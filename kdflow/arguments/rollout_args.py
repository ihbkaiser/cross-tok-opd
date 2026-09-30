from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RolloutArguments:
    """ Arguments for rollout (on-policy distillation)."""
    
    rollout_disable_piecewise_cuda_graph: bool = field(default=False, metadata={"help": "Disable experimental SGLang piecewise graph capture for incompatible models."})
    rollout_deterministic_inference: bool = field(default=False, metadata={"help": "Enable SGLang deterministic inference (seeded sampler) in the rollout serving processes."})
    rollout_random_seed: int = field(default=-1, metadata={"help": "Server-level random seed for deterministic rollout serving; -1 keeps the SGLang default."})
    rollout_attention_backend: str = field(default="", metadata={"help": "Explicit SGLang attention backend for rollout serving; empty keeps the model default. Deterministic inference requires a supported backend, so it defaults to flashinfer."})
    rollout_disable_radix_cache: bool = field(default=False, metadata={"help": "Force the SGLang radix cache off for rollout serving. Deterministic serving pins it off so backends stay comparable."})
    enforce_max_sequence_length: bool = field(default=False, metadata={"help": "Bound sampled response plus prompt and terminal sentinel by data.max_len."})
    exact_token_trajectory: bool = field(default=False, metadata={"help": "Text-only sampled-ID and teacher-ID contract."})
    diagnostic_max_updates: int = field(default=0, metadata={"help": "Stop after this many updates without changing the scheduler horizon."})
    diagnostic_collapse_gate: bool = field(default=False, metadata={"help": "Stop after two collapsed rollout batches."})

    rollout_num_engines: int = field(
        default=0,
        metadata={"help": "The number of engines for rollout."}
    )
    rollout_tp_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size for each vLLM engine."}
    )
    rollout_enable_sleep: bool = field(
        default=False,
        metadata={"help": "Enable sleep mode for vLLM."}
    )
    rollout_mem_fraction_static: float = field(
        default=0.6,
        metadata={"help": "GPU memory utilization for each vLLM engine."}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling for rollout."}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for rollout."}
    )
    n_samples_per_prompt: int = field(
        default=1,
        metadata={"help": "Sample n responses per prompt."}
    )
    rollout_batch_size: int = field(
        default=32,
        metadata={"help": "Number of prompts for each rollout."}
    )
    generate_max_len: int = field(
        default=2048,
        metadata={"help": "Max generation tokens during rollout."}
    )
    print_rollout_sample: bool = field(
        default=False,
        metadata={"help": "Whether to print a rollout sample after each rollout."}
    )
    
    