from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig

from kdflow.utils import get_tokenizer
from kdflow.models.ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor


def forward_position_ids(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Position ids for the non-packing forward path.

    HF's eager attention takes a different path when ``position_ids`` is passed, and
    on a right-padded batch that path moves the longest row's logits by up to 15.5
    relative to the same row forwarded alone. Measured on the MP-OPD parity capture
    ``failure-zusi05xj``: that capture's longest row is 4096 tokens = the full padded
    width, so its mask is all ones and its ``cumsum - 1`` *is* ``arange``; passing it
    is therefore meant to be a no-op, yet it changed the logits. The same capture
    showed the padded forward reproducing the trainer's stored logprobs bit-for-bit
    while a per-row forward reproduced SGLang's, i.e. this path is what made the
    behaviour/trainer parity guard fire.

    Right-padding needs no explicit positions: HF's default ``cache_position``
    already numbers each row ``0..n-1`` over its real tokens and the padded tail is
    masked. So return None, which reproduces the per-row (and SGLang) result. Keep
    the explicit form only for masks that are not a plain prefix, where HF's default
    would be wrong.
    """
    if attention_mask is None:
        return None
    mask = attention_mask.long()
    # Right padding means the ones are exactly a prefix of each row. Comparing the
    # mask against a prefix template built from its own row sums tests that directly.
    # A cumsum-based test does not work: cumsum keeps increasing through the padded
    # tail, and clamping it to 1 yields all ones, which never equals the mask. The
    # regression tests caught exactly that mistake.
    ones = mask.sum(-1, keepdim=True)
    template = (torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < ones).long()
    if bool(template.eq(mask).all()):
        return None
    position_ids = mask.cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 1)
    return position_ids


class DistillModel(nn.Module):
    """
    Base class for student models in knowledge distillation (modified from OpenRLHF/openrlhf/models/actor.py).

    Args:
        args (Arguments): Arguments.
        strategy (Strategy): Strategy for student model loading and training.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
    """

    def __init__(
        self,
        strategy,
        device_map=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.temperature = self.args.rollout.temperature
        model_name_or_path = self.args.model.student_name_or_path

        # Support multiple attention mechanism implementations
        attn_impl = self.args.model.attn_implementation

        self.model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # Determine if this is a Vision-Language model
        self.is_vl_model = hasattr(self.model_config, "vision_config")
        
        if self.is_vl_model:
            model_class = AutoModelForImageTextToText
        elif self.args.model.use_liger_kernel:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            model_class = AutoLigerKernelForCausalLM
        else:
            model_class = AutoModelForCausalLM

        if hasattr(self.model_config, "text_config"):
            self.hidden_size = self.model_config.text_config.hidden_size
        else:
            self.hidden_size = self.model_config.hidden_size
        
        self.model = strategy.load_hf_model(
            model_class, 
            model_name_or_path, 
            attn_impl, 
            self.model_config, 
        )
        
        # LoRA
        if self.args.model.lora_rank > 0:
            # https://github.com/huggingface/peft/issues/137
            self.model.enable_input_require_grads()
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.args.model.lora_rank,
                lora_alpha=self.args.model.lora_alpha,
                target_modules=self.args.model.target_modules,
                lora_dropout=self.args.model.lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.tokenizer = get_tokenizer(model_name_or_path, self.model)

        # https://github.com/huggingface/transformers/issues/26877
        # Use `model.generate(use_cache=True)` instead.`
        self.model.config.use_cache = False

        # packing samples using Flash Attention 2
        self.packing_samples = self.args.data.packing_samples
        
        self._print_model()

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        allgather_logits=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Returns action log probs"""
        batch, seqlen = sequences.size()
        foward_attention_mask = attention_mask
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            foward_attention_mask = None
        else:
            position_ids = forward_position_ids(attention_mask)

        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids, output_hidden_states=output_hidden_states, **kwargs)
        
        if allgather_logits and self.packing_samples:
            output["logits"] = gather_and_pad_tensor(
                output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
            ).squeeze(-2)
            if output_hidden_states and "hidden_states" in output:
                output["hidden_states"] = list(output["hidden_states"])
                for i in range(len(output["hidden_states"])):
                     output["hidden_states"][i] = gather_and_pad_tensor(
                        output["hidden_states"][i], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                    ).squeeze(-2)
        return output

    def _print_model(self):
        self.strategy.print(f"Student Model: \n  {self.model}")
    
    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": self.args.train.gradient_checkpointing_use_reentrant
            }
        )

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
