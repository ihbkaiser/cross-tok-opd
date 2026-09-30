import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class DistillationArguments:
    """ Arguments for knowledge distillation."""
    
    kd_ratio: float = field(
        default=0.5,
        metadata={"help": "Loss = (1 - kd_ratio) * nll_loss + kd_ratio * kd_loss."}
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for knowledge distillation."}
    )
    kd_algorithm: str = field(
        default="vanilla_kd",
        metadata={"help": "KD algorithm for each training step."}
    )
    kd_loss_fn: str = field(
        default="kl",
        metadata={"help": "Divergence selection for knowledge distillation, e.g., kl, rkl, js."}
    )
    teacher_forward_n_batches: int = field(
        default=1,
        metadata={"help": "Teacher forward N global batches at once for student multi-step training."}
    )
    teacher_enable_sleep: bool = field(
        default=False,
        metadata={"help": "Sleep teacher when not needed."}
    )
    teacher_offload_tags: str = field(
        default="all",
        metadata={"help": "Offload tags for sglang."}
    )
    teacher_quantization: str = field(
        default=None
    )
    teacher_tp_size: int = field(
        default=8,
        metadata={"help": "Tensor parallel size for teacher model."}
    )
    teacher_ep_size: int = field(
        default=1,
        metadata={"help": "Expert parallel size for teacher model (only for MoE models)."}
    )
    teacher_pp_size: int = field(
        default=1,
        metadata={"help": "Pipeline parallel size for teacher model."}
    )
    teacher_dp_size: int = field(
        default=1,
        metadata={"help": "Data parallel size for teacher model."}
    )
    teacher_mem_fraction_static: float = field(
        default=0.4,
        metadata={"help": "Memory fraction for teacher model."}
    )
    teacher_context_length: Optional[int] = field(
        default=None,
        metadata={"help": "Context length for teacher model. If None, use model's default max_position_embeddings."}
    )
    teacher_update_freq: int = field(
        default=1,
        metadata={"help": "Weight update frequency for teacher model."}
    )
    span_score_mode: str = field(default="raw_logit", metadata={"choices": ["raw_logit", "mean_logprob"], "help": "Span scoring: released code or paper Eq.7."})
    # DSKD hyperparameters
    dskd_token_align: str = field(
        default="eta",
        metadata={
            "help": "Token alignment strategy for cross-tokenizer DSKD. Options: 'cma' (cross-model attention), 'eta' (exact token alignment).", 
            "choices": ["eta", "cma"]
        }
    )
    dskd_topk_vocab: int = field(
        default=-1,
        metadata={"help": "Number of top vocabulary tokens used for projector initialization. -1 means using all tokens."}
    )
    dskd_projector_lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for DSKD projectors."}
    )
    # JSD
    jsd_beta: float = field(
        default=0.5,
        metadata={"help": "Beta for Jensen-Shannon Divergence."}
    )
    # Skewed KL/RKL
    skew_lambda: float = field(
        default=0.1,
        metadata={"help": "Lambda for Skewed KL/RKL."}
    )
    # Adaptive KL
    adaptive_alpha: float = field(
        default=0.5,
        metadata={"help": "Alpha for Adaptive KL Divergence."}
    )
    # Hierarchical Ranking Loss
    hrl_topk: int = field(
        default=5,
        metadata={"help": "Top-k Ranking for Hierarchical Ranking Loss."}
    )
    # ALM (Approximate Likelihood Matching) hyperparameters
    alm_temperature: float = field(
        default=100.0,
        metadata={"help": "Temperature tau for ALM binarised f-divergence. Higher values focus more on longer/lower-likelihood chunks."}
    )
    alm_f_divergence: str = field(
        default="kl",
        metadata={"help": "f-divergence function for ALM. Options: 'kl' (KL-divergence), 'tvd' (Total Variation Distance)."}
    )
    alm_debiasing: bool = field(
        default=False,
        metadata={"help": "Enable outcome chunk debiasing for ALM."}
    )
    alm_debiasing_threshold: float = field(
        default=0.1,
        metadata={"help": "Threshold gamma for ALM outcome chunk debiasing."}
    )
    # ULD (Universal Logit Distillation) hyperparameters
    uld_lambda: float = field(
        default=1.5,
        metadata={"help": "Lambda weight for ULD Wasserstein-1 distance loss."}
    )
    uld_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for ULD softmax computation."}
    )
    uld_top_k: int = field(
        default=1024,
        metadata={"help": "Top-k approximation for ULD Wasserstein-1 distance. "
                  "Only the top-k largest probabilities are kept to reduce memory. "
                  "Set to -1 to disable (use full vocabulary). Default 1024."}
    )
    # Random Span ablation (simple_ctkd_random_span)
    random_span_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of aligned tokens to be randomly merged into spans "
                  "for the simple_ctkd_random_span ablation. 0.0 means no merging "
                  "(equivalent to simple_ctkd). Range: [0.0, 1.0]."}
    )
    # Span mask ablation (span_ctkd_no_span_loss)
    span_mask_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of span segments whose loss is masked (set to zero) "
                  "in the span_ctkd_no_span_loss ablation. 1.0 means all span loss "
                  "is masked (original behaviour); 0.0 means no masking (all spans "
                  "contribute to loss). Range: [0.0, 1.0]."}
    )
    # SimCT (span_ctkd) re-normalization safeguard: mask positions whose
    # re-normalization multiplier G(h) = 1 / Z_T(h) exceeds this threshold,
    # where Z_T(h) is the teacher mass captured by the SimCT candidate space.
    # G(h) > threshold means the candidate space holds < 1/threshold of the
    # teacher mass, so re-normalization would amplify a low-mass tail; such
    # positions contribute no distillation gradient. Set to 0 to disable.
    span_gh_mask_threshold: float = field(
        default=2.0,
        metadata={"help": "Mask the distillation loss at positions where the SimCT "
                  "re-normalization multiplier G(h)=1/Z_T(h) exceeds this value "
                  "(default 2.0, i.e. teacher captured mass Z_T(h) < 0.5). "
                  "Set to 0 to disable masking."}
    )
    # NVIDIA X-Token P-KL projection settings. The projection artifact is
    # model-pair-specific and therefore must be supplied explicitly.
    xtoken_projection_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the audited X-Token top-k projection artifact."},
    )
    xtoken_projection_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Required SHA-256 for xtoken_projection_path."},
    )
    xtoken_vocab_topk: int = field(
        default=8192,
        metadata={"help": "Teacher-vocabulary top-k used by X-Token P-KL."},
    )
    xtoken_exact_match_only: bool = field(
        default=False,
        metadata={"help": "Only use canonically exact alignment chunks for X-Token P-KL."},
    )
    xtoken_dynamic_loss_scaling: bool = field(
        default=True,
        metadata={"help": "Dynamically scale X-Token KD magnitude to student CE."},
    )
    xtoken_kl_loss_weight: float = field(default=1.0)
    xtoken_ce_loss_scale: float = field(default=0.1)
    xtoken_max_comb_len: int = field(default=4)

    # MP-OPD scalar canonical-path credit and contiguous atom partitioning.
    mp_opd_mode: str = field(
        default="atomic",
        metadata={"choices": ["atomic", "fixed", "random", "oracle", "soft"]},
    )
    mp_opd_max_span_length: int = field(default=4)
    mp_opd_min_span_length: int = field(default=1)
    mp_opd_fixed_span_length: int = field(default=2)
    mp_opd_partition_temperature: float = field(default=1.0)
    mp_opd_random_seed: int = field(default=43)
    mp_opd_energy_hidden_dim: int = field(default=32)
    mp_opd_energy_layers: int = field(default=2)
    mp_opd_energy_lr: float = field(default=1e-3)
    mp_opd_energy_checkpoint: Optional[str] = field(default=None)
    mp_opd_alternating: bool = field(default=False)
    mp_opd_meta_path: Optional[str] = field(default=None)
    mp_opd_meta_batch_size: int = field(default=16)
    mp_opd_meta_microbatch_size: int = field(default=4)
    mp_opd_energy_every: int = field(default=1)
    mp_opd_offload_adam_moments: bool = field(
        default=False,
        metadata={"help": "Diagnostic: offload student Adam moments during full-meta mixed VJP."},
    )
    mp_opd_host_mask: bool = field(
        default=False,
        metadata={"help": "Candidate-only: copy semi-Markov boolean mask metadata to host once per partition."},
    )

    def __post_init__(self):
        # Validate teacher parallel size settings
        if self.teacher_ep_size > self.teacher_tp_size:
            raise ValueError(
                f"SGLang requires that teacher_ep_size ({self.teacher_ep_size}) must be <= teacher_tp_size ({self.teacher_tp_size}). "
            )
        if self.teacher_tp_size % self.teacher_ep_size != 0:
            raise ValueError(
                f"SGLang requires that teacher_tp_size ({self.teacher_tp_size}) must be divisible by teacher_ep_size ({self.teacher_ep_size})."
            )
        # Validate KD hyperparameters
        if not 0.0 <= self.kd_ratio <= 1.0:
            raise ValueError(f"kd_ratio must be in [0, 1], got {self.kd_ratio}.")
        if self.kd_temperature <= 0:
            raise ValueError(f"kd_temperature must be > 0, got {self.kd_temperature}.")
        if not 0.0 < self.teacher_mem_fraction_static <= 1.0:
            raise ValueError(f"teacher_mem_fraction_static must be in (0, 1], got {self.teacher_mem_fraction_static}.")
        if not 0.0 <= self.random_span_ratio <= 1.0:
            raise ValueError(f"random_span_ratio must be in [0, 1], got {self.random_span_ratio}.")
        if not 0.0 <= self.span_mask_ratio <= 1.0:
            raise ValueError(f"span_mask_ratio must be in [0, 1], got {self.span_mask_ratio}.")
        if self.kd_algorithm == "xtoken":
            if not self.xtoken_projection_path:
                raise ValueError("xtoken_projection_path is required for kd_algorithm=xtoken.")
            if not self.xtoken_projection_sha256 or len(self.xtoken_projection_sha256) != 64:
                raise ValueError(
                    "a 64-character xtoken_projection_sha256 is required for kd_algorithm=xtoken."
                )
            if self.xtoken_vocab_topk <= 0:
                raise ValueError("xtoken_vocab_topk must be positive.")
            if self.xtoken_max_comb_len <= 0:
                raise ValueError("xtoken_max_comb_len must be positive.")
        if self.kd_algorithm == "mp_opd":
            if self.mp_opd_mode not in {"atomic", "fixed", "random", "oracle", "soft"}:
                raise ValueError(f"unsupported mp_opd_mode: {self.mp_opd_mode}")
            if self.mp_opd_max_span_length <= 0 or self.mp_opd_fixed_span_length <= 0:
                raise ValueError("MP-OPD span lengths must be positive")
            if not 1 <= self.mp_opd_min_span_length <= self.mp_opd_max_span_length:
                raise ValueError("mp_opd_min_span_length must be at least 1 and no larger "
                                 "than mp_opd_max_span_length")
            if self.mp_opd_partition_temperature <= 0:
                raise ValueError("mp_opd_partition_temperature must be positive")
            if self.mp_opd_energy_hidden_dim <= 0 or self.mp_opd_energy_layers <= 0:
                raise ValueError("MP-OPD energy dimensions must be positive")
            if self.mp_opd_energy_lr <= 0:
                raise ValueError("mp_opd_energy_lr must be positive")
            if self.mp_opd_mode == "soft" and not self.mp_opd_energy_checkpoint:
                raise ValueError("mp_opd_energy_checkpoint is required for soft mode")
