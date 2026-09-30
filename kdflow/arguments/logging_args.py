from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoggingArguments:
    """ Arguments for logging (e.g., wandb and tensorboard)."""

    logging_steps: int = field(
        default=10,
        metadata={"help": "Log results every n steps."}
    )
    use_tensorboard: bool = field(
        default=False,
        metadata={"help": "Write the same scalar metrics to TensorBoard event files."},
    )
    tensorboard_log_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "TensorBoard event directory. Defaults to "
                "<save_path>/tensorboard when TensorBoard is enabled."
            )
        },
    )
    tensorboard_flush_secs: int = field(
        default=10,
        metadata={"help": "Maximum TensorBoard background flush interval in seconds."},
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Use wandb for logging."}
    )
    wandb_org: str = field(
        default=None
    )
    wandb_project: str = field(
        default=None
    )
    wandb_group: str = field(
        default=None
    )
    wandb_run_name: str = field(
        default=None
    )
    wandb_run_id: str = field(
        default=None,
        metadata={"help": "Stable W&B run id; enables exact post-run verification."},
    )
    wandb_job_type: str = field(
        default="train",
    )
    wandb_tags: str = field(
        default="",
        metadata={"help": "Comma-separated W&B tags."},
    )
    wandb_mode: str = field(
        default="online",
        metadata={"help": "wandb mode: online, offline, or disabled."}
    )
    wandb_dir: str = field(
        default=None,
        metadata={"help": "Directory to store wandb offline logs."}
    )
