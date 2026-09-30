import os
import time
import json
import math
import subprocess
import threading
from tqdm import tqdm
from datetime import timedelta
from dataclasses import asdict
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from kdflow.trajectory import trajectory_tokens, collapse_observation
from kdflow.deadline import stop_before_rollout
from kdflow.datasets.utils import get_tokenizer_or_processor
from kdflow.utils.logging_utils import init_logger
from kdflow.utils.tensorboard_utils import create_tensorboard_logger
from kdflow.utils.utils import zero_pad_sequences


logger = init_logger(__name__)


def progress_metrics(*, current_updates, session_start_updates, session_elapsed_seconds,
                     campaign_target_updates, diagnostic_target_updates):
    """Return cumulative and session-scoped progress/ETA values."""
    current = max(0, int(current_updates))
    session_start = max(0, int(session_start_updates))
    campaign_target = max(0, int(campaign_target_updates))
    diagnostic_target = max(0, int(diagnostic_target_updates))
    session_completed = max(0, current - session_start)
    elapsed = max(0.0, float(session_elapsed_seconds))
    def ratio(target):
        return min(1.0, max(0.0, current / target)) if target else 0.0
    def eta(target):
        if target <= 0 or current >= target:
            return 0.0
        if session_completed <= 0 or elapsed <= 0:
            return None
        return (target - current) * elapsed / session_completed
    return {
        "current_optimizer_updates": current,
        "session_start_optimizer_updates": session_start,
        "session_completed_updates": session_completed,
        "session_fit_seconds": elapsed,
        "campaign_target_updates": campaign_target,
        "diagnostic_stop_target": diagnostic_target,
        "scientific_total_updates": campaign_target,
        "campaign_progress": ratio(campaign_target),
        "diagnostic_progress": ratio(diagnostic_target),
        "scientific_progress": ratio(campaign_target),
        "eta_campaign_seconds": eta(campaign_target),
        "eta_diagnostic_seconds": eta(diagnostic_target),
    }


def format_eta(seconds):
    return "unknown" if seconds is None else str(timedelta(seconds=int(max(0.0, seconds)))).split(".")[0]


class OnPolicyKDTrainer:
    """
    Ray-based trainer for on-policy knowledge distillation.
    """
    
    def __init__(
        self,
        strategy,
        student_model,
        teacher_model,
        rollout_group,
        is_same_tokenizer: bool,
        train_dataloader,
        eval_dataloader=None,
        max_rollout_iters: int = None,
        num_rollout_iters_per_epoch: int = None,
        generate_kwargs: Dict[str, float] = None,
        meta_sampler=None,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            strategy: Training strategy containing configuration
            student_model: StudentActorGroup
            teacher_model: TeacherActorGroup
            rollout_group: RolloutGroup
            is_same_tokenizer: Whether student and teacher use same tokenizer
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            max_rollout_iters: Maximum rollout iterations in training
            num_rollout_iters_per_epoch: Number of rollout iterations per epoch
        """
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher = teacher_model
        self.rollout_group = rollout_group
        self.is_same_tokenizer = is_same_tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_rollout_iters = max_rollout_iters
        self.num_rollout_iters_per_epoch = num_rollout_iters_per_epoch
        self.generate_kwargs = generate_kwargs
        self.meta_sampler = meta_sampler
        self.epochs = self.args.train.num_epochs
        
        self.image_key = getattr(self.args.data, "image_key", None)
        self.student_processor = get_tokenizer_or_processor(
            self.args.model.student_name_or_path,
            need_processor=self.image_key is not None,
        )
        self.teacher_processor = None
        if self.args.model.teacher_name_or_path:
            if not self.is_same_tokenizer:
                self.teacher_processor = get_tokenizer_or_processor(
                    self.args.model.teacher_name_or_path,
                    need_processor=self.image_key is not None,
                )
            else:
                self.teacher_processor = self.student_processor
        
        self.world_size = self.args.train.num_nodes * self.args.train.num_gpus_per_node
        
        assert self.args.kd.kd_ratio == 1.0, "On-policy KD only supports kd_ratio=1.0."
        
        self.log_state = defaultdict(list)
        self.completed_optimizer_updates = 0
        self._resource_stop = threading.Event()
        self._resource_thread = None
        self._resource_sample_index = 0
        from kdflow.training_checkpoint import pipeline_contract, inspect
        self._checkpoint_contract = pipeline_contract(self.args, len(train_dataloader.dataset))
        self._resume_directory = None
        checkpoint_root = Path(self.args.train.ckpt_path)
        if self.args.train.load_checkpoint:
            if self.args.model.student_name_or_path == self.args.model.teacher_name_or_path:
                raise ValueError("Self-distillation resume requires the lagged teacher snapshot; unsupported")
            self._resume_directory, _ = inspect(checkpoint_root, self._checkpoint_contract, self.world_size)
        elif (checkpoint_root / "latest.json").exists():
            raise ValueError("Training checkpoint exists; explicitly enable load_checkpoint to resume")
        self._extra_checkpoint_steps = {int(x) for x in self.args.train.resume_checkpoint_steps.split(",") if x.strip()}
        if any(x <= 0 for x in self._extra_checkpoint_steps) or self.args.train.pause_after_updates < 0:
            raise ValueError("Checkpoint/pause steps must be positive")
        self._init_loggers()
    
    def _init_loggers(self) -> None:
        """Initialize optional experiment loggers."""
        self._wandb = None
        self._tensorboard = create_tensorboard_logger(
            self.args,
            default_log_dir=os.path.join(self.args.train.save_path, "tensorboard"),
        )
        
        if self.args.log.use_wandb:
            import wandb

            if not self.args.log.wandb_run_id:
                raise ValueError("wandb_run_id is required when use_wandb=True")
            
            self._wandb = wandb
            if self.args.log.wandb_mode != "offline" and not wandb.api.api_key:
                wandb.login()
            wandb.init(
                entity=self.args.log.wandb_org,
                project=self.args.log.wandb_project,
                group=self.args.log.wandb_group,
                name=self.args.log.wandb_run_name,
                id=self.args.log.wandb_run_id,
                resume="allow" if self.args.train.load_checkpoint else "never",
                job_type=self.args.log.wandb_job_type,
                tags=[tag.strip() for tag in self.args.log.wandb_tags.split(",") if tag.strip()],
                config=asdict(self.args),
                reinit=True,
                mode=self.args.log.wandb_mode,
                dir=self.args.log.wandb_dir,
            )
            
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
            wandb.define_metric("system/sample_index")
            wandb.define_metric("system/*", step_metric="system/sample_index", step_sync=True)

    def _resource_logging_loop(self) -> None:
        """Continuously publish GPU telemetry while the training loop is active."""
        while not self._resource_stop.is_set():
            try:
                sampled = self._sample_gpu_resources()
                payload = {
                    "system/sample_index": self._resource_sample_index,
                    "system/wall_time_seconds": time.time() - self.start_time,
                }
                payload.update({f"system/{key}": value for key, value in sampled.items()})
                if self._wandb is not None:
                    self._wandb.log(payload)
                if self._tensorboard is not None:
                    self._tensorboard.log(payload, step=self._resource_sample_index)
                self._resource_sample_index += 1
            except Exception as error:
                logger.warning("GPU resource sampler failed: %s", error)
            self._resource_stop.wait(15)

    def _start_resource_logger(self) -> None:
        if (
            (self._wandb is None and self._tensorboard is None)
            or self._resource_thread is not None
        ):
            return
        self._resource_thread = threading.Thread(
            target=self._resource_logging_loop,
            name="experiment-gpu-resource-sampler",
            daemon=True,
        )
        self._resource_thread.start()

    def _stop_resource_logger(self) -> None:
        self._resource_stop.set()
        if self._resource_thread is not None:
            self._resource_thread.join(timeout=20)

    @staticmethod
    def _sample_gpu_resources() -> Dict[str, float]:
        """Return non-sensitive point-in-time NVIDIA metrics for every visible GPU."""
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        metrics: Dict[str, float] = {}
        for line in result.stdout.splitlines():
            index, used, total, util, power, temperature = [part.strip() for part in line.split(",")]
            prefix = f"resource/gpu_{index}"
            metrics[f"{prefix}/memory_used_mib"] = float(used)
            metrics[f"{prefix}/memory_total_mib"] = float(total)
            metrics[f"{prefix}/utilization_percent"] = float(util)
            metrics[f"{prefix}/power_watts"] = float(power)
            metrics[f"{prefix}/temperature_c"] = float(temperature)
        return metrics
    
    def _print_training_config(self) -> None:
        """Log training configuration before training starts."""
        total_steps = self.max_rollout_iters
        grad_accum = self.args.train.train_batch_size * self.args.model.ring_attn_size \
            // (self.args.train.micro_train_batch_size * self.args.train.num_nodes * self.args.train.num_gpus_per_node)
        
        logger.info("******* Start Training *******")
        logger.info(f"  Num Epochs:            {self.epochs}")
        logger.info(f"  Steps per Epoch:       {self.num_rollout_iters_per_epoch}")
        logger.info(f"  Total Training Steps:  {total_steps}")
        logger.info(f"  Per-device Batch Size: {self.args.train.micro_train_batch_size}")
        logger.info(f"  Gradient Accumulation: {grad_accum}")
        logger.info(f"  Learning Rate:         {self.args.train.learning_rate}")
        logger.info(f"  KD Algorithm:          {self.args.kd.kd_algorithm}")
        logger.info(f"  KD Loss Function:      {self.args.kd.kd_loss_fn}")
    
    def _save_training_checkpoint(self, epoch, consumed_samples):
        from kdflow.training_checkpoint import begin, publish, capture_rng, prune_complete
        boundary = (self.completed_optimizer_updates, epoch, consumed_samples)
        if getattr(self, "_last_saved_boundary", None) == boundary:
            return
        directory = begin(self.args.train.ckpt_path, self.completed_optimizer_updates)
        ranks = ray.get([a.save_training_state.remote(str(directory)) for a in self.student._actor_handlers])
        client = {"epoch": epoch, "consumed_samples": consumed_samples,
                  "global_step": self.global_step, "optimizer_updates": self.completed_optimizer_updates,
                  "energy_updates": self.completed_energy_updates,
                  "collapse_baseline": self._collapse_baseline, "collapse_streak": self._collapse_streak,
                  "resource_sample_index": self._resource_sample_index, "rng": capture_rng()}
        publish(self.args.train.ckpt_path, directory, step=self.completed_optimizer_updates,
                world_size=self.world_size, contract=self._checkpoint_contract, client=client, rank_files=ranks)
        self._last_saved_boundary = boundary
        prune_complete(self.args.train.ckpt_path)
        return directory

    def fit(self, global_step=0, start_epoch=0):
        import random
        import numpy as np
        random.seed(self.args.train.seed)
        np.random.seed(self.args.train.seed)
        torch.manual_seed(self.args.train.seed)
        self.global_step = global_step
        diagnostic_limit = getattr(self.args.rollout, "diagnostic_max_updates", 0)
        if diagnostic_limit < 0:
            raise ValueError("diagnostic_max_updates must be nonnegative")
        expected_updates = min(self.max_rollout_iters, diagnostic_limit) if diagnostic_limit else self.max_rollout_iters
        self._collapse_baseline = None
        self._collapse_streak = 0
        self._collapse_stop = False
        self.stop_reason = None
        previous_step_seconds = 0.0
        consumed_samples = 0
        self.completed_energy_updates = None
        restored_rng = None
        if self._resume_directory:
            client = torch.load(self._resume_directory / "driver.pt", map_location="cpu", weights_only=False)
            self.global_step = client["global_step"]
            self.completed_optimizer_updates = client["optimizer_updates"]
            self.completed_energy_updates = client["energy_updates"]
            start_epoch = client["epoch"]
            consumed_samples = client["consumed_samples"]
            self._collapse_baseline = client["collapse_baseline"]
            self._collapse_streak = client["collapse_streak"]
            self._resource_sample_index = client["resource_sample_index"]
            restored_rng = client["rng"]
            if self.args.train.enable_sleep:
                self.student.wakeup()
            ray.get([a.restore_training_state.remote(str(self._resume_directory)) for a in self.student._actor_handlers])
        
        # Print training configuration and initialize loggers
        self._print_training_config()

        # Create Gloo IPC groups between training ranks and rollout engines (following slime)
        rollout_tp_size = getattr(self.args.rollout, "rollout_tp_size", 1)
        self.student.connect_rollout_engines(self.rollout_group.actors, rollout_tp_size)
        if self.args.model.student_name_or_path == self.args.model.teacher_name_or_path:   # for self-distillation
            num_gpus_per_teacher_actor = self.args.kd.teacher_tp_size * self.args.kd.teacher_pp_size
            self.student.connect_teacher_actors(self.teacher.teacher_engines, num_gpus_per_teacher_actor)
        if self._resume_directory:
            # Fresh serving processes initially contain the SFT weights. Restore
            # their student snapshot before admitting any new generation.
            if self.args.train.enable_sleep:
                self.rollout_group.wakeup(tags=["weights"])
            self.student.update_rollout_weights()
            if self.args.train.enable_sleep:
                self.rollout_group.sleep(tags=["weights"])
                self.student.sleep()
            if self.args.model.student_name_or_path == self.args.model.teacher_name_or_path:
                raise ValueError("Self-distillation resume needs a saved lagged teacher snapshot; unsupported")
        if restored_rng:
            from kdflow.training_checkpoint import restore_rng
            restore_rng(restored_rng)
        
        self.session_start_optimizer_updates = int(self.completed_optimizer_updates)
        self.session_start_time = time.time()
        self.start_time = self.session_start_time
        self.campaign_target_updates = int(self.max_rollout_iters)
        self.diagnostic_stop_target = int(expected_updates)
        self._start_resource_logger()
        num_micro_batches = self.args.train.train_batch_size // self.args.train.micro_train_batch_size
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_consumed = consumed_samples if epoch == start_epoch else 0
            self.train_dataloader.sampler.set_epoch(epoch, consumed_samples=epoch_consumed)
            # DataLoader iterator creation must not advance the driver RNG after
            # a restart. PromptDataset is deterministic; order comes from sampler.
            self.train_dataloader.generator = torch.Generator().manual_seed(self.args.train.seed + epoch)
            
            for prompt_batch in self.train_dataloader:
                if self.completed_optimizer_updates >= expected_updates:
                    break
                if stop_before_rollout(time.time(), previous_step_seconds):
                    self.stop_reason = "deadline_checkpoint_reserve"
                    break
                self.global_step += 1
                meta_rows=(self.meta_sampler.batch(self.global_step,[x['stu_prompt'] for x in prompt_batch])
                           if self.meta_sampler else None)
                step_started = time.time()
                
                rollout_start = time.time()
                rollout_samples = self.rollout(prompt_batch, **self.generate_kwargs)
                rollout_time = time.time() - rollout_start

                self.log_state["rollout_time"].append(rollout_time)
                if self._collapse_stop:
                    self.stop_reason = self.stop_reason or "collapse_gate_before_update"
                    self.logging()
                    break

                teacher_start = time.time()
                if self.args.train.enable_sleep:
                    self.teacher.wakeup()
                    
                rollout_samples_for_kd = self.teacher.forward(rollout_samples)
                
                if self.args.train.enable_sleep:
                    self.teacher.sleep()
                self.log_state["teacher_fwd_time"].append(time.time() - teacher_start)
                
                all_global_batches = []
                for i in range(0, len(rollout_samples), num_micro_batches):
                    global_batch = rollout_samples_for_kd[i : i + num_micro_batches]
                    
                    global_batch_token_num = sum(mb["stu_loss_mask"].sum() for mb in global_batch)
                    avg_micro_batch_token_num = global_batch_token_num / len(global_batch)
                    for mb in global_batch:
                        mb["avg_micro_batch_token_num"] = avg_micro_batch_token_num
                    all_global_batches.append(global_batch)
                
                student_start = time.time()
                
                if self.args.train.enable_sleep:
                    self.student.wakeup()
                
                for global_batch in all_global_batches:
                    status_list = ray.get(self.student.async_run_distill(global_batch,meta_rows))
                    for k in status_list[0].keys():
                        self.log_state[k].append(sum(s[k] for s in status_list) / len(status_list))
                    optimizer_updates = [int(round(s["optimizer_updates"])) for s in status_list]
                    if len(set(optimizer_updates)) != 1 or optimizer_updates[0] != 1:
                        raise RuntimeError(
                            f"expected exactly one optimizer update per rollout iteration, got {optimizer_updates}"
                        )
                    self.completed_optimizer_updates += optimizer_updates[0]
                    if 'mp_opd_energy_updates_total' in status_list[0]:
                        self.completed_energy_updates = int(round(status_list[0]['mp_opd_energy_updates_total']))
                        
                self.log_state["student_train_time"].append(time.time() - student_start)
                
                ray.get([actor.empty_cache.remote() for actor in self.student._actor_handlers])

                # update weights in rollout actors
                if self.args.train.enable_sleep:
                    self.rollout_group.wakeup(tags=["weights"])
                update_start = time.time()
                self.student.update_rollout_weights()
                self.log_state["weight_update_time"].append(time.time() - update_start)
                if self.args.train.enable_sleep:
                    self.rollout_group.sleep(tags=["weights"])
                
                # update weights in teacher actors (only for self-distillation)
                if self.args.model.teacher_name_or_path == self.args.model.student_name_or_path \
                    and self.global_step % self.args.kd.teacher_update_freq == 0:
                    if self.args.train.enable_sleep:
                        self.teacher.wakeup(tags=["weights"])
                    teacher_update_start = time.time()
                    self.student.update_teacher_weights()
                    self.log_state["teacher_update_time"].append(time.time() - teacher_update_start)
                    if self.args.train.enable_sleep:
                        self.teacher.sleep(tags=["weights"])
                    
                # Save checkpoint BEFORE sleep so FSDP2 model is still on GPU
                # (avoids DTensor issues after CPU offload/reload)
                checkpoint_due = ((self.args.train.save_steps > 0 and self.global_step % self.args.train.save_steps == 0)
                    or self.completed_optimizer_updates in self._extra_checkpoint_steps)
                if checkpoint_due:
                    self.strategy.log(f"Saving model at global step {self.global_step}")
                    save_path = os.path.join(self.args.train.save_path, f"step{self.global_step}")
                    ray.get(self.student.async_save_model(save_path))

                if self.args.train.enable_sleep:
                    self.student.sleep()

                step_wall_time = time.time() - step_started
                previous_step_seconds = step_wall_time
                self.log_state["step_wall_time"].append(step_wall_time)
                self.log_state["completed_optimizer_updates"].append(
                    float(self.completed_optimizer_updates)
                )
                self.log_state["rollout_prompt_count"].append(float(len(prompt_batch)))
                self.log_state["rollout_sample_count"].append(
                    float(len(prompt_batch) * self.args.rollout.n_samples_per_prompt)
                )
                valid_tokens = self.log_state.get("global_processed_valid_student_tokens", [])
                if valid_tokens:
                    self.log_state["end_to_end_valid_tokens_per_second"].append(
                        float(valid_tokens[-1]) / step_wall_time if step_wall_time > 0 else 0.0
                    )
                for key, value in self._sample_gpu_resources().items():
                    if math.isfinite(value):
                        self.log_state[key].append(value)
                    
                self.logging()
                epoch_consumed += len(prompt_batch)
                pause_due = (self.args.train.pause_after_updates > 0
                             and self.completed_optimizer_updates >= self.args.train.pause_after_updates)
                if checkpoint_due or pause_due:
                    if self.args.train.enable_sleep:
                        self.student.wakeup()
                    self._save_training_checkpoint(epoch, epoch_consumed)
                    if checkpoint_due:
                        from kdflow.export_ready import publish as publish_export
                        publish_export(self.args.train.save_path, self.completed_optimizer_updates)
                    if self.args.train.enable_sleep:
                        self.student.sleep()
                if pause_due:
                    self.stop_reason = "checkpoint_pause"
                    break
        
            # Save model after each epoch
            # Note: student is already in sleep state after the last step's sleep() call,
            # so we need to wakeup before saving
            self.strategy.log(f"Saving model after epoch {epoch + 1}")
            save_path = os.path.join(self.args.train.save_path, f"step{self.completed_optimizer_updates}")
            if self.args.train.enable_sleep:
                self.student.wakeup()
            from kdflow.export_ready import marker, publish as publish_export
            # A published export may already be read by another host. Never
            # rewrite it at the epoch boundary (notably step156 and step312).
            already_exported = marker(self.args.train.save_path, self.completed_optimizer_updates).exists()
            if not already_exported:
                ray.get(self.student.async_save_model(save_path))
            if not self._collapse_stop:
                self._save_training_checkpoint(epoch, epoch_consumed)
                if not already_exported:
                    publish_export(self.args.train.save_path, self.completed_optimizer_updates)
            if self.args.train.enable_sleep:
                self.student.sleep()
            if self.stop_reason or self.completed_optimizer_updates >= expected_updates:
                break

        total_time = time.time() - self.session_start_time
        final_progress = progress_metrics(current_updates=self.completed_optimizer_updates,
            session_start_updates=self.session_start_optimizer_updates,
            session_elapsed_seconds=total_time,
            campaign_target_updates=self.campaign_target_updates,
            diagnostic_target_updates=self.diagnostic_stop_target)
        self.strategy.log(f"Training done, totally cost {str(timedelta(seconds=total_time)).split('.')[0]}")

        if not self.stop_reason and self.completed_optimizer_updates != expected_updates:
            raise RuntimeError(
                f"optimizer update gate failed: {self.completed_optimizer_updates} != {expected_updates}"
            )

        summary_path = os.path.join(self.args.train.save_path, "run-summary.json")
        os.makedirs(self.args.train.save_path, exist_ok=True)
        with open(summary_path + ".pending", "w") as handle:
            json.dump(
                {
                    "status": "stopped" if self.stop_reason else "completed",
                    "stop_reason": self.stop_reason,
                    "rollout_iterations": self.global_step,
                    "optimizer_updates": self.completed_optimizer_updates,
                    "energy_updates": self.completed_energy_updates,
                    "total_time_seconds": total_time,
                    **final_progress,
                    "kd_algorithm": self.args.kd.kd_algorithm,
                    "student": self.args.model.student_name_or_path,
                    "teacher": self.args.model.teacher_name_or_path,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        os.replace(summary_path + ".pending", summary_path)

        if self._wandb is not None or self._tensorboard is not None:
            self._stop_resource_logger()
        if self._wandb is not None:
            self._wandb.run.summary["optimizer_updates"] = self.completed_optimizer_updates
            self._wandb.run.summary["rollout_iterations"] = self.global_step
            self._wandb.run.summary["training_completed"] = not bool(self.stop_reason)
            self._wandb.run.summary["stop_reason"] = self.stop_reason or ("diagnostic_limit_reached" if diagnostic_limit else "completed")
            self._wandb.run.summary["total_time_seconds"] = total_time
            self._wandb.run.summary["resource_samples"] = self._resource_sample_index
            for key, value in final_progress.items():
                self._wandb.run.summary[key] = value
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.log(
                {
                    "summary/optimizer_updates": self.completed_optimizer_updates,
                    "summary/rollout_iterations": self.global_step,
                    "summary/training_completed": int(not bool(self.stop_reason)),
                    "summary/total_time_seconds": total_time,
                    "summary/resource_samples": self._resource_sample_index,
                    **{f"summary/{key}": (0.0 if value is None else value)
                       for key, value in final_progress.items()},
                },
                step=self.global_step,
            )
            self._tensorboard.close()
            
    def rollout(self, prompt_batch: List[Dict[str, str]], **kwargs) -> List[dict]:
        """Generate samples using rollout engine.

        Args:
            prompt_batch: List of dicts with keys: datasource, stu_prompt, tea_prompt, label
            **kwargs: Additional arguments for generation

        Returns:
            List of rollout sample dicts containing generated samples
        """
        if self.args.train.enable_sleep:
            self.rollout_group.wakeup()

        # Extract prompts and labels from batch
        all_stu_prompts = [item["stu_prompt"] for item in prompt_batch]
        all_tea_prompts = [item["tea_prompt"] for item in prompt_batch]
        all_labels = [item["label"] for item in prompt_batch]
        all_images = [item.get("images") for item in prompt_batch] if self.image_key else None
        
        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = self.args.rollout.n_samples_per_prompt
        all_stu_prompts = sum([[p] * n_samples_per_prompt for p in all_stu_prompts], [])
        all_tea_prompts = sum([[p] * n_samples_per_prompt for p in all_tea_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        if all_images:
            all_images = sum([[imgs] * n_samples_per_prompt for imgs in all_images], [])
        
        exact = getattr(self.args.rollout, "exact_token_trajectory", False)
        prompt_ids = None
        if exact:
            if all_images:
                raise ValueError("explicit trajectory mode is text-only")
            tok = getattr(self.student_processor, "tokenizer", self.student_processor)
            prompt_ids = [self._encode_prompt_ids(tok, p) for p in all_stu_prompts]
        sampling_params = self.generate_kwargs
        if getattr(self.args.rollout, "enforce_max_sequence_length", False):
            if prompt_ids is None:raise ValueError("Sequence cap requires exact token trajectory")
            from kdflow.trajectory import bounded_sampling_params
            sampling_params = bounded_sampling_params(prompt_ids, sampling_params, self.args.data.max_len)
        from kdflow.training_checkpoint import seeded_sampling
        sampling_params = seeded_sampling(sampling_params, seed=self.args.train.seed,
                                          step=self.global_step, count=len(all_stu_prompts))
        all_outputs = self.rollout_group.generate(all_stu_prompts, sampling_params, image_data=all_images, input_ids=prompt_ids)

        rollout_dir = os.path.join(self.args.train.save_path, "rollout_data")
        os.makedirs(rollout_dir, exist_ok=True)
        with open(os.path.join(rollout_dir, f"{self.global_step}.jsonl"), "w") as f:
            for prompt, output in zip(all_stu_prompts, all_outputs):
                record = {"prompt": prompt, "output": output["text"]}
                if exact:
                    record.update(prompt_ids=output["prompt_ids"], output_ids=output["output_ids"],
                                  meta_info=output.get("meta_info", {}),
                                  behavior_weight_version=self.completed_optimizer_updates)
                if "reward_result" in output:
                    record["reward_result"] = output["reward_result"]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Process outputs into rollout samples
        sample_list = [
            self._build_rollout_sample(
                stu_prompt=all_stu_prompts[i],
                tea_prompt=all_tea_prompts[i],
                output=all_outputs[i],
                label=all_labels[i],
                images=all_images[i] if all_images and all_images[i] else None,
            )
            for i in range(len(all_outputs))
        ]
        
        if getattr(self.args.rollout, "diagnostic_collapse_gate", False):
            eos = getattr(self.student_processor, "tokenizer", self.student_processor).eos_token_id
            lengths = [len(o["output_ids"]) - int(bool(o["output_ids"]) and o["output_ids"][-1] == eos) for o in all_outputs]
            observation = collapse_observation(lengths, self._collapse_baseline, self._collapse_streak)
            self._collapse_baseline = observation["content_length_baseline"]
            self._collapse_streak = observation["collapse_bad_streak"]
            self._collapse_stop = observation["collapse_stop"]
            for key, value in observation.items():
                self.log_state[key].append(float(value))

        if exact and self.args.kd.kd_algorithm == "mp_opd":
            from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer
            stu_tok = getattr(self.student_processor, "tokenizer", self.student_processor)
            tea_tok = getattr(self.teacher_processor, "tokenizer", self.teacher_processor)
            atomizer = SimCTAtomizer(stu_tok, tea_tok)
            valid = 0
            for index, sample in enumerate(sample_list):
                stu_labels = sample["stu_input_ids"].roll(-1)[sample["stu_loss_mask"]].tolist()
                tea_labels = sample["tea_input_ids"].roll(-1)[sample["tea_loss_mask"]].tolist()
                valid += int(atomizer.atomize(stu_labels, tea_labels, sample_id=str(index)).valid)
            self.log_state["trajectory_valid_sample_count"].append(float(valid))
            if valid == 0:
                self._collapse_stop = True
                self.stop_reason = "zero_valid_samples_before_update"

        # Print sample for debugging
        sample0 = sample_list[0]["stu_prompts"][0] + sample_list[0]["stu_responses"][0]
        if self.args.rollout.print_rollout_sample:
            print(sample0)
        
        micro_batch_list = self._collate_micro_batches(sample_list, self.args.train.micro_train_batch_size)
        
        if self.args.train.enable_sleep:
            self.rollout_group.sleep()

        return micro_batch_list
    
    @staticmethod
    def _collate_values(key: str, values: list):
        """Collate a list of per-sample values into a single micro-batch value.
        
        Rules:
        - mm_* tensors       → torch.cat (variable patch counts across images)
        - scalar tensors     → torch.cat (metrics like response_length)
        - sequence tensors   → zero_pad_sequences (pad variable-length sequences)
        - lists              → flatten
        - None               → keep None
        - scalars            → collect into list
        """
        v0 = values[0]
        if isinstance(v0, torch.Tensor):
            if key.startswith("mm_"):
                return torch.cat(values, dim=0)
            return zero_pad_sequences(values, side="right", value=0)
        if isinstance(v0, list):
            return sum(values, [])
        if v0 is None:
            return None
        return values

    def _collate_micro_batches(self, sample_list: List[Dict], batch_size: int) -> List[Dict]:
        """Collate single samples into micro-batches."""
        micro_batch_list = []
        for i in range(0, len(sample_list), batch_size):
            batch_samples = sample_list[i : i + batch_size]
            micro_batch = {
                key: self._collate_values(key, [s[key] for s in batch_samples])
                for key in batch_samples[0]
            }
            micro_batch_list.append(micro_batch)
        return micro_batch_list

    def _tokenize_sample(
        self, 
        prompt: str, 
        response: str, 
        processor,
        prefix: str,
        images=None,
    ) -> Dict[str, Any]:
        """Tokenize prompt + response for a single sample.

        Args:
            prompt: Chat-templated prompt string.
            response: Response string.
            processor: Processor or tokenizer for the model.
            prefix: 'stu' or 'tea'.
            images: PIL images (or None for text-only).

        Returns:
            Dict with ``{prefix}_input_ids``, ``{prefix}_attn_mask``, ``{prefix}_loss_mask``
            and optional multimodal fields.
        """
        tokenizer = getattr(processor, "tokenizer", processor)
        resp_tok = tokenizer(response, return_tensors="pt", add_special_tokens=False)
        resp_len = resp_tok["input_ids"].shape[1]

        full_input = {"text": prompt + response}
        if images:
            full_input["images"] = images
        full_tok = processor(**full_input, return_tensors="pt", add_special_tokens=False)
        prompt_len = full_tok["input_ids"].shape[1] - resp_len

        # since rollout response does not contain eos token, we need to add it manually
        eos_token_id = tokenizer.eos_token_id
        input_ids = torch.cat([full_tok["input_ids"][0], full_tok["input_ids"][0].new_tensor([eos_token_id])])
        attn_mask = torch.cat([full_tok["attention_mask"][0], full_tok["attention_mask"][0].new_ones(1)])
        loss_mask = torch.tensor(
            [False] * (prompt_len - 1) + [True] * (resp_len + 1) + [False],
            device=input_ids.device,
        )

        result = {
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attn_mask": attn_mask,
            f"{prefix}_loss_mask": loss_mask,
        }
        # Extract multimodal fields (e.g., pixel_values, image_grid_thw)
        for k, v in full_tok.items():
            if k not in ("input_ids", "attention_mask"):
                v = torch.as_tensor(v)
                result[f"mm_{k}"] = v.squeeze(0) if v.dim() > 2 else v

        return result

    def _build_rollout_sample(
        self,
        stu_prompt: str,
        tea_prompt: str,
        output,
        label: str,
        images=None,
    ) -> Dict[str, Any]:
        """
        Build a single rollout sample with both student and teacher tokenizations.
        
        Args:
            stu_prompt: Student prompt string (formatted with student's chat template)
            tea_prompt: Teacher prompt string (formatted with teacher's chat template)
            output: rollout output object
            label: Label string
            images: PIL images (or None for text-only).
            
        Returns:
            Dict containing all sample fields
        """
        if getattr(self.args.rollout, "exact_token_trajectory", False):
            return self._build_exact_rollout_sample(stu_prompt, tea_prompt, output, label, images)
        # Decode response using student tokenizer
        response_ids = output["output_ids"]
        response_text = output["text"]
        
        stu_tokens = self._tokenize_sample(
            stu_prompt, response_text, self.student_processor, "stu", images=images
        )
        
        if not self.is_same_tokenizer or tea_prompt != stu_prompt:
            tea_tokens = self._tokenize_sample(
                tea_prompt, response_text, self.teacher_processor, "tea", images=images
            )
        else:
            # Same tokenizer: reuse student tensors (mm_ fields already in stu_tokens)
            tea_tokens = {
                "tea_input_ids": stu_tokens["stu_input_ids"].clone(),
                "tea_attn_mask": stu_tokens["stu_attn_mask"].clone(),
                "tea_loss_mask": stu_tokens["stu_loss_mask"].clone(),
            }
        
        response_length = len(response_ids)
        total_length = stu_tokens["stu_attn_mask"].float().sum()
        
        # Build tea_full_text for teacher actor (SGLang engine uses raw text)
        tokenizer = getattr(self.teacher_processor, "tokenizer", self.teacher_processor)
        tea_full_text = tea_prompt + response_text + " " + tokenizer.eos_token

        sample = {
            **tea_tokens,
            **stu_tokens,
            "tea_full_texts": [tea_full_text],
            "rollout_log_probs": None,
            "stu_prompts": [stu_prompt],
            "stu_responses": [response_text],
            "tea_prompts": [tea_prompt],
            "labels": [label],
            "response_length": torch.FloatTensor([[response_length]]),
            "total_length": torch.FloatTensor([[total_length]]),
        }
        if images:
            sample["images"] = [images]
        return sample
            
    def _encode_prompt_ids(self, tokenizer, prompt):
        # Rendered HF chat templates already contain their BOS/control tokens.
        data_args = getattr(getattr(self, "args", None), "data", None)
        rendered = getattr(data_args, "apply_chat_template", False)
        return tokenizer(prompt, add_special_tokens=not rendered)["input_ids"]

    def _build_exact_rollout_sample(self, stu_prompt, tea_prompt, output, label, images):
        if images:
            raise ValueError("explicit trajectory mode is text-only")
        stu_tok = getattr(self.student_processor, "tokenizer", self.student_processor)
        tea_tok = getattr(self.teacher_processor, "tokenizer", self.teacher_processor)
        sampled = output["output_ids"]
        content_ids = sampled[:-1] if sampled and sampled[-1] == stu_tok.eos_token_id else sampled
        kd_args = getattr(getattr(self, "args", None), "kd", None)
        if getattr(kd_args, "kd_algorithm", None) == "mp_opd":
            from kdflow.algorithms._mp_opd_atoms import mp_content_ids
            content_ids, _ = mp_content_ids(sampled, stu_tok)
        response = stu_tok.decode(content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        expected_prompt = self._encode_prompt_ids(stu_tok, stu_prompt)
        if output["prompt_ids"] != expected_prompt:
            raise RuntimeError("rollout/trainer prompt ID mismatch")
        stu_ids, stu_mask, synthetic = trajectory_tokens(output["prompt_ids"], sampled, stu_tok.eos_token_id)
        tea_prompt_ids = self._encode_prompt_ids(tea_tok, tea_prompt)
        tea_response_ids = tea_tok(response, add_special_tokens=False)["input_ids"]
        tea_ids, tea_mask, _ = trajectory_tokens(tea_prompt_ids, tea_response_ids, tea_tok.eos_token_id)
        result = {}
        for prefix, ids, mask in [("stu", stu_ids, stu_mask), ("tea", tea_ids, tea_mask)]:
            result[prefix + "_input_ids"] = torch.tensor(ids, dtype=torch.long)
            result[prefix + "_attn_mask"] = torch.ones(len(ids), dtype=torch.long)
            result[prefix + "_loss_mask"] = torch.tensor(mask, dtype=torch.bool)
        logprobs = output.get("meta_info", {}).get("output_token_logprobs")
        if logprobs is None or [int(x[1]) for x in logprobs] != sampled:
            raise RuntimeError("behavior log-prob IDs do not match sampled IDs")
        if any(not math.isfinite(float(x[0])) for x in logprobs):
            raise RuntimeError("non-finite sampled behavior log-probability")
        behavior = torch.full((len(stu_ids),), float("nan"))
        start = len(expected_prompt) - 1
        behavior[start:start + len(sampled)] = torch.tensor([float(x[0]) for x in logprobs])
        result.update(
            tea_full_texts=[tea_prompt + response], rollout_log_probs=None,
            stu_behavior_log_probs=behavior,
            stu_prompts=[stu_prompt], stu_responses=[response], tea_prompts=[tea_prompt], labels=[label],
            response_length=torch.FloatTensor([[len(sampled)]]),
            total_length=torch.FloatTensor([[len(stu_ids)]]),
        )
        return result

    def logging(self):
        if self.global_step % self.args.log.logging_steps == 0:
            now = time.time()
            session_start = getattr(self, "session_start_optimizer_updates", 0)
            current_updates = getattr(self, "completed_optimizer_updates", 0)
            session_start_time = getattr(self, "session_start_time", getattr(self, "start_time", now))
            fallback_target = getattr(self, "max_rollout_iters", self.num_rollout_iters_per_epoch * self.epochs)
            campaign_target = getattr(self, "campaign_target_updates", fallback_target)
            diagnostic_target = getattr(self, "diagnostic_stop_target", campaign_target)
            metrics = progress_metrics(current_updates=current_updates,
                session_start_updates=session_start,
                session_elapsed_seconds=now - session_start_time,
                campaign_target_updates=campaign_target,
                diagnostic_target_updates=diagnostic_target)
            progress_str = ("epoch [{current_epoch}/{total_epoch}], "
                "step [{current_step}/{total_step}], "
                "optimizer_updates [{updates}/{campaign_target}], "
                "train_progress [{campaign_progress:.2f}%], "
                "campaign_progress [{campaign_progress:.2f}%], "
                "diagnostic_progress [{diagnostic_progress:.2f}%], "
                "Elapsed: {elapsed}, ETA: {eta_campaign}, "
                "ETA_campaign: {eta_campaign}, ETA_diagnostic: {eta_diagnostic}, ").format(
                current_epoch=self.current_epoch + 1, total_epoch=self.epochs,
                current_step=self.global_step,
                total_step=self.num_rollout_iters_per_epoch * self.epochs,
                updates=metrics["current_optimizer_updates"],
                campaign_target=metrics["campaign_target_updates"],
                campaign_progress=metrics["campaign_progress"] * 100,
                diagnostic_progress=metrics["diagnostic_progress"] * 100,
                elapsed=str(timedelta(seconds=int(metrics["session_fit_seconds"]))).split(".")[0],
                eta_campaign=format_eta(metrics["eta_campaign_seconds"]),
                eta_diagnostic=format_eta(metrics["eta_diagnostic_seconds"]))
            active_log_state = {}
            for k, value in self.log_state.items():
                if isinstance(value, list):
                    if not value:
                        continue
                    value = sum(value) / len(value)
                active_log_state[k] = value
            log_info = []
            for k, value in active_log_state.items():
                if k in {"lr", "optimizer_lr_used", "mp_opd_energy_parameter_delta",
                         "mp_opd_energy_gradient_norm", "mp_opd_virtual_gradient_norm"}:
                    log_info.append(f"{k}: {value:.9e}")
                else:
                    log_info.append(f"{k}: {value:.6f}")
            self.strategy.log(progress_str + ", ".join(log_info))
            if self._wandb is not None or self._tensorboard is not None:
                logs = {"train/global_step": self.global_step}
                for key, value in metrics.items():
                    if value is not None:
                        logs[f"train/{key}"] = value
                for k, value in active_log_state.items():
                    logs[f"train/{k}"] = value
            if self._wandb is not None:
                self._wandb.log(logs)
            if self._tensorboard is not None:
                self._tensorboard.log(logs, step=self.global_step)
            self.log_state.clear()

