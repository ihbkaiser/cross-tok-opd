"""Frozen contract for the one-shot Modal X-Token Phase A run."""

APP_NAME = "vdt-xtoken-phase-a-no1ceboy-20260904-r14"
VOLUME_NAME = "vdt-xtoken-phase-a-no1ceboy-20260904-r14"
SECRET_NAME = "vdt-xtoken-hf-no1ceboy"
RUN_ID = "xtoken-phase-a-a10x2-10step-20260904-r14"

GPU = "A10:2"
GPU_COUNT = 2
CONTAINER_MEMORY_MIB = 131_072
EPHEMERAL_DISK_MIB = 524_288
OPTIMIZER_UPDATES = 10
GLOBAL_BATCH = 64
SEQUENCE_LENGTH = 2048
MICRO_BATCH = 1
PRECISION = "bfloat16"

STUDENT_REPO = "meta-llama/Llama-3.2-1B"
STUDENT_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
STUDENT_WEIGHT_SHA256 = "68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a"
TEACHER_REPO = "Qwen/Qwen3-4B"
TEACHER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
NEMO_REVISION = "13a10647ebbf0f940d2b06ea41800b3f2fb46099"
DATA_REVISION = "13fa979be2e7f7e62913eee0ec5e97c8fd6e24af"
NATIVE_LOCK_SHA256 = "145d512cf6e56deec88eacfde4159ba97fd55496a26e26d5aec8d33b7ba357cb"

TARGET_NAME = "target-4b-2048-b64-full-10steps-modal-a10x2-r14"
OVERLAY_NAME = "NeMo-RL-cpu-logits-modal-a10x2-r14"


def scientific_contract():
    return {
        "algorithm": "upstream X-Token off-policy P-KL+CE",
        "student": {"repo": STUDENT_REPO, "revision": STUDENT_REVISION},
        "teacher": {"repo": TEACHER_REPO, "revision": TEACHER_REVISION},
        "nemo_revision": NEMO_REVISION,
        "data_revision": DATA_REVISION,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "global_batch": GLOBAL_BATCH,
        "micro_batch": MICRO_BATCH,
        "sequence_length": SEQUENCE_LENGTH,
        "precision": PRECISION,
        "projection": "top32 reverse+scale+special -> exact remap -> top4",
        "optimizer": "upstream AdamW 5e-5",
        "scheduler": "upstream warmup250/cosine4750",
        "seed": 42,
        "checkpointing": False,
        "wandb": False,
        "scope": "bounded engineering workload; fixed-corpus off-policy, not OPD",
    }


def operational_contract():
    return {
        "profile": "no1ceboy",
        "gpu": GPU,
        "gpu_count": GPU_COUNT,
        "container_memory_mib": CONTAINER_MEMORY_MIB,
        "ephemeral_disk_mib": EPHEMERAL_DISK_MIB,
        "nccl_cumem_host_enable": "0",
        "backend": "dense HF/SDPA FSDP2",
        "teacher_logit_transport": (
            "same-node CPU memfd, exact FP32 zero-low16 packing, "
            "microbatch-only GPU upload"
        ),
        "automatic_retry": False,
    }
