import os

import ray
import torch
import numpy as np

from kdflow.utils.utils import remove_pad_token
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


def chunk_batch_indices(batch_indices, chunk_size):
    """Split assigned micro-batch indices without changing their order."""
    chunk_size = max(1, int(chunk_size))
    return [
        batch_indices[start : start + chunk_size]
        for start in range(0, len(batch_indices), chunk_size)
    ]


@ray.remote
class TeacherRayActor:
    """
    TeacherRayActor: Responsible for teacher model forward (prefilling) using SGLang Engine.
    
    Key design: Teacher and Student SHARE the same GPUs via PlacementGroup co-location.
    - TeacherRayActor is scheduled on PG bundles with RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    - GPU binding is managed via base_gpu_id parameter passed to SGLang Engine
    - This allows Teacher and Student to share the same GPUs via PlacementGroup
    """
    
    def __init__(self, strategy, base_gpu_id: int = 0, nnodes: int = 1, node_rank: int = 0, dist_init_addr: str = None):
        """
        Initialize TeacherRayActor.
        
        Args:
            strategy: Training strategy containing configuration args
            base_gpu_id: Base GPU device ID for SGLang Engine binding (e.g., 0, 1, 2, ...)
                        Used with RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES for PG co-location.
            nnodes: Number of nodes for multi-node tp/pp
            node_rank: Rank of this node in tp+pp group
            dist_init_addr: Address for distributed initialization
        """
        # Keep module import lightweight for CPU utility tests. The production
        # engine dependency is required exactly when an actor is instantiated.
        from kdflow.backend.sglang.sglang_engine import SGLangEngineService, EngineConfig

        logger.info(f"[TeacherRayActor] __init__ STARTED, PID={os.getpid()}, base_gpu_id={base_gpu_id}")
        
        self.strategy = strategy
        self.tp_size = strategy.args.kd.teacher_tp_size
        self.ep_size = strategy.args.kd.teacher_ep_size
        self.pp_size = strategy.args.kd.teacher_pp_size
        self.base_gpu_id = base_gpu_id
        self.node_rank = node_rank
        
        # Disable tokenizers parallelism to avoid deadlock with multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Create engine configuration
        # GPU binding is handled by base_gpu_id (works with RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES)
        self.engine_config = EngineConfig(
            model_path=strategy.args.model.teacher_name_or_path,
            tp_size=self.tp_size,
            ep_size=self.ep_size,
            pp_size=self.pp_size,
            chunked_prefill_size=-1,  # Disable chunked prefill for full sequence processing
            disable_radix_cache=True,  # Disable cache for deterministic behavior
            enable_return_hidden_states=True,  # Enable hidden states extraction
            enable_memory_saver=True,  # Enable memory saving mode
            enable_weights_cpu_backup=True,  # Backup weights to CPU for memory release
            quantization=strategy.args.kd.teacher_quantization,
            mem_fraction_static=strategy.args.kd.teacher_mem_fraction_static,
            context_length=strategy.args.kd.teacher_context_length,
            offload_tags=strategy.args.kd.teacher_offload_tags,
            base_gpu_id=self.base_gpu_id,
            nnodes=nnodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
        )
        
        # Initialize SGLang Engine service (runs in subprocess)
        self.engine_service = SGLangEngineService(self.engine_config)
        self.engine_service.start()
        
        if self.strategy.args.train.enable_sleep and self.node_rank == 0:
            logger.info(f"[TeacherRayActor] Teacher sleep after initialization")
            self.engine_service.sleep(tags=self.strategy.args.kd.teacher_offload_tags)
        
        logger.info(f"[TeacherRayActor] Initialized with tp_size={self.tp_size}, ep_size={self.ep_size}, pp_size={self.pp_size}")

    def ready(self):
        """Return True when the actor is ready (engine service started)."""
        return self.engine_service._started

    def forward(self, global_batch, batch_indices):
        """
        Perform forward pass (prefilling) on the given batches.
        
        Args:
            global_batch: List of all micro-batches
            batch_indices: List of batch indices this actor should process
        
        Returns:
            List of (batch_idx, micro-batch with teacher_hiddens) tuples and return timestamp
        """
        chunk_size = max(1, int(self.strategy.args.kd.teacher_forward_n_batches))
        results_with_indices = []  # List of (original_batch_idx, batch_with_hiddens)
        index_chunks = chunk_batch_indices(batch_indices, chunk_size)
        num_chunks = len(index_chunks)
        for chunk_number, current_indices in enumerate(index_chunks, start=1):
            batches = [global_batch[i] for i in current_indices]
            mbsz = batches[0]["tea_input_ids"].shape[0]

            # Bound each SGLang request so long cross-tokenizer sequences do not
            # create one monolithic prefill/hidden-state transfer.
            prompts = sum((micro_batch["tea_full_texts"] for micro_batch in batches), [])
            unpadded_loss_masks = []
            for micro_batch in batches:
                attn_mask, loss_mask = micro_batch["tea_attn_mask"], micro_batch["tea_loss_mask"]
                unpadded_loss_masks.extend(remove_pad_token(loss_mask, attn_mask, return_tensors=True))
            unpadded_loss_masks = [m.numpy().astype(bool) for m in unpadded_loss_masks]

            image_data = None
            if batches[0].get("images") is not None:
                image_data = sum((micro_batch["images"] for micro_batch in batches), [])

            logger.info(
                "[TeacherRayActor.forward] chunk %d/%d, micro_batches=%d, samples=%d",
                chunk_number,
                num_chunks,
                len(batches),
                len(prompts),
            )
            exact_ids = None
            if getattr(self.strategy.args.rollout, "exact_token_trajectory", False):
                if image_data is not None:
                    raise ValueError("explicit trajectory mode is text-only")
                exact_ids = []
                for mb in batches:
                    exact_ids.extend([x.tolist() for x in remove_pad_token(mb["tea_input_ids"], mb["tea_attn_mask"], return_tensors=True)])
            hidden_states_list = self.engine_service.generate(
                prompt=prompts,
                loss_masks=unpadded_loss_masks,
                sampling_params={"max_new_tokens": 0},
                return_hidden_states=True,
                image_data=image_data,
                input_ids=exact_ids,
            )

            sample_idx = 0
            for mb_idx, original_batch_idx in enumerate(current_indices):
                mb_hidden_np = hidden_states_list[sample_idx : sample_idx + mbsz]
                mb_hidden_np = np.concatenate(mb_hidden_np, axis=0)
                batches[mb_idx]["teacher_hiddens"] = ray.put(mb_hidden_np)
                results_with_indices.append((original_batch_idx, batches[mb_idx]))
                sample_idx += mbsz
        
        return results_with_indices
    
    def sleep(self, tags=None):
        """Release GPU memory occupation, move weights to CPU."""
        if tags is None:
            tags = self.strategy.args.kd.teacher_offload_tags
        self.engine_service.sleep(tags=tags)
        
    def wakeup(self, tags=None):
        """Resume GPU memory occupation, move weights back to GPU."""
        if tags is None:
            tags = self.strategy.args.kd.teacher_offload_tags
        self.engine_service.wakeup(tags=tags)
        
    def update_weights_from_tensor(self, serialized_named_tensors, load_format, flush_cache):
        return self.engine_service.update_weights_from_tensor(
            serialized_named_tensors, load_format, flush_cache)
    
    def flush_cache(self):
        """Flush cache. No-op for teacher engine since disable_radix_cache=True."""
        pass
    
    def shutdown(self):
        """Shutdown the engine service."""
        self.engine_service.shutdown()
        logger.info("[TeacherRayActor] Shutdown complete")
