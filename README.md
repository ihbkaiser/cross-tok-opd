# SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation

This repository contains the code for reproducing the experiments in our paper *"SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation"*.

## MP-OPD trajectory diagnostic

The experimental full-student alternating campaign and shared on-policy
training-state resume contract are documented in
[Full alternating and pipeline resume](docs/full_alternating_resume.md).
Its six long runs are gated by a target-GPU pause/resume qualification;
CPU tests alone do not establish FSDP2 higher-order or B200 memory support.

For the bounded r7-collapse follow-up, see
[the corrected-trajectory diagnostic](docs/mp_opd_path_diagnostic.md).

## Fork experiment: X-Token native uv

The `vdt/experiments/xtoken` branch adds an isolated
[X-Token setup and diagnostic harness](experiments/xtoken/README.md) using a
commit-pinned NVIDIA NeMo-RL checkout and its own uv environment. It does not
change the root KDFlow/SimCT dependencies or algorithms. The separate
[native CUDA 12.8 projects](experiments/environments/README.md) contain actual
checked-in `pyproject.toml`/`uv.lock` pairs for **both X-Token and SimCT**.
The earlier two-3090 host completed a [three-update X-Token training smoke](experiments/environments/reports/2026-09-03-2x3090.md)
with finite metrics, after an opt-in NCCL cuMem host-memory workaround.
SimCT has passed environment/CUDA canaries but has not trained. This is
**off-policy engineering evidence**, not OPD or reproduction of either paper's
quality results.

The [new 70-GiB host / larger-workload report](experiments/environments/reports/2026-09-03-70gb-target.md)
tracks the separately requested Llama-1B <- Qwen3-4B, sequence-2048,
global-batch-64 ten-update test. Its setup and training outcome must not be
inferred from the earlier small-smoke result.

For B200 deployment on a company server, use the
[restricted-network installation runbook](experiments/environments/B200_INSTALL.md).
It covers the public-lock path, a private internal-package-pool overlay, exact
local model paths, CUDA/uv gates, SFT qualification, SimCT qualification, W&B,
and failure diagnostics without committing company endpoints or credentials.
The same runbook documents the portable
[`Dockerfile.b200-cu130`](docker/Dockerfile.b200-cu130), which contains the two
isolated locked B200 environments and all source/launch code while requiring
models, datasets, checkpoints, TensorBoard events, and W&B outputs to be
mounted from outside the image. It can be built locally or pushed to Docker Hub
from the guarded Modal VM-Sandbox wrapper.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Reproducing Experiments](#reproducing-experiments)
  - [Step 1: Data Preparation](#step-1-data-preparation)
  - [Step 2: Generate Teacher Responses](#step-2-generate-teacher-responses)
  - [Step 3: SFT Warmup Training](#step-3-sft-warmup-training)
  - [Step 4: Cross-Tokenizer On-Policy Distillation](#step-4-cross-tokenizer-on-policy-distillation)
  - [Step 5: Evaluation](#step-5-evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

---

## Overview

**Experimental Setup:**

| Role | Model |
|------|-------|
| Teacher | Qwen2.5-7B-Instruct |
| Teacher | Phi-4-mini-instruct |
| Student | Gemma-2-2B-IT |
| Student | Phi-4-mini-instruct |

**Evaluation Benchmarks:** GSM8K, MATH-500, MBPP, LiveCodeBench-v6

---

## Environment Setup

Our experiments are built on top of [KDFlow](https://github.com/songmzhang/KDFlow). Please install the dependencies:

```bash
git clone https://github.com/sunjie279/SimCT-.git
cd SimCT_
pip install -e ./
pip install flash_attn==2.8.3 --no-build-isolation
```

Then set the following environment variables:

```bash
export MODEL_PATH="./models"         # Directory containing model weights
export DATA_PATH="./data"            # Directory for datasets
export OUTPUT_PATH="./output/ckpts"  # Directory for checkpoints
```

Required model weights (download from HuggingFace):
- `$MODEL_PATH/Qwen2.5-7B-Instruct`
- `$MODEL_PATH/Phi-4-mini-instruct`
- `$MODEL_PATH/gemma-2-2b-it`

---

## Reproducing Experiments

The full pipeline consists of 5 steps:

```
Data Preparation → Generate Teacher Responses → SFT Warmup → Distillation Training → Evaluation
```

### Step 1: Data Preparation

We construct a 10K mixed math+code training dataset from multiple sources:

```bash
# Download raw datasets from HuggingFace
python scripts/data/download_datasets.py

# Prepare individual datasets
python scripts/data/prepare_gsm8k.py
python scripts/data/prepare_orca_math.py

# Build the 10K mixed dataset
python scripts/data/prepare_mixed_math_code.py
```

This produces:
- `$DATA_PATH/mixed_math_code_10k/` — Training prompts
- `$DATA_PATH/mixed_math_code_10k_with_source/` — Training prompts with source labels

### Step 2: Generate Teacher Responses

Generate 8 trajectories per question for each teacher model using SGLang:

```bash
# Qwen2.5-7B-Instruct
bash scripts/sft/run_generate_responses_10k_qwen.sh

# Phi-4-mini-instruct
bash scripts/sft/run_generate_responses_10k_phi4.sh
```

Each script starts an SGLang server (DP=8), generates responses (temperature=0.6, top_p=0.95), and saves to `$DATA_PATH/teacher_responses_10k_<model_tag>/`.

### Step 3: SFT Warmup Training

Before distillation, the student needs an SFT warmup to establish basic instruction-following capability.

#### 3.1 Build SFT Dataset

Select the shortest correct response per question from teacher trajectories:

```bash
# From Qwen responses
bash scripts/sft/run_build_sft_10k_qwen.sh

# From Phi-4 responses
bash scripts/sft/run_build_sft_10k_phi4.sh
```

#### 3.2 Run SFT Training

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for SFT:

```bash
# Gemma-2-2B-IT with Qwen teacher data
bash scripts/sft/run_gemma2_sft_warmup_10k_qwen.sh

# Gemma-2-2B-IT with Phi-4 teacher data
bash scripts/sft/run_gemma2_sft_warmup_10k_phi4.sh

# Phi-4-mini with Qwen teacher data
bash scripts/sft/run_phi4_sft_warmup_10k_qwen.sh
```

### Step 4: Cross-Tokenizer On-Policy Distillation

Run distillation training for each teacher→student pair:

```bash
# Qwen2.5-7B → Gemma-2-2B
bash scripts/ctopd/qwen25_gemma2_span_mix10k_lr5e-7.sh

# Qwen2.5-7B → Phi-4-mini
bash scripts/ctopd/qwen25_phi4_span_mix10k_lr5e-7.sh

# Phi-4-mini → Gemma-2-2B
bash scripts/ctopd/phi4_gemma2_span_mix10k_lr5e-7.sh
```


### Step 5: Evaluation

For the pinned author-code contract and the separately labelled paper-spec
profile, use [the contract runner](scripts/evaluation/CONTRACT_README.md).
It adds offline data preparation, explicit seeds, result provenance and sandboxed
scoring without starting GPU servers. The legacy commands below describe the
original workflow and are not a verified RunAI deployment recipe.

Evaluate on GSM8K, MATH-500, MBPP, and LiveCodeBench-v6:

```bash
# Prepare LiveCodeBench data (one-time)
python scripts/evaluation/prepare_lcb_data.py

# Run all evaluations
bash scripts/evaluation/eval_all_monitor.sh
```

Or evaluate a single checkpoint:

```bash
python scripts/evaluation/evaluation.py \
    --model_path $MODEL_PATH/your-checkpoint \
    --dataset gsm8k \
    --base_url http://127.0.0.1:30000 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 1 \
    --max_tokens 4096
```

Supported datasets: `gsm8k`, `math500`, `mbpp`, `live-code-bench-v6`

---

## Acknowledgement

This codebase is built on top of [KDFlow](https://github.com/songmzhang/KDFlow), a user-friendly and efficient framework for LLM knowledge distillation. We sincerely thank the KDFlow team for their excellent work.

---

## Citation

```bibtex
@article{sun2026simctrecoveringlostsupervision,
      title={SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation}, 
      author={Jie Sun and Mao Zheng and Mingyang Song and Qiyong Zhong and Yilin Cheng and Bichuan Feng and Pengfei Liu and Junfeng Fang and Xiang Wang},
      year={2026},
      eprint={2605.07711},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.07711}, 
}
```
# RunAI evaluation default

For new contract evaluations, use separated generation/scoring: 256 concurrent
generation requests per GPU (`eval_queue.py worker`, default phase `generate`)
and independent CPU `score-spool` workers on any node sharing the plan. The borrowed campaign
creates both jobs automatically. See `experiments/runai/BORROWED_8H.md` for the
contract and migration constraints. This does not reconfigure running workers.

The shared pool protocol and worker commands are documented in
`experiments/runai/SHARED_EVAL_POOL.md`.
