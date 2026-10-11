# Qwen3 student headroom screen

`pair_screen_modal.py` reuses the pinned B200 runtime and stores model/data on a
Modal Volume. It never installs dependencies or downloads model weights locally.
`prepare` pins current HF revisions, verifies test-set sizes, saves file hashes,
and selects the same 128 MATH-500 and 128 GSM8K items for both student modes.

Run these on a host with the authorized Modal CLI profile, not on the company node:

```bash
export PAIR_SCREEN_RUN="llama-qwen-screen-$(date -u +%Y%m%dT%H%M%S)"
modal run --profile kieusontung6 experiments/modal/pair_screen_modal.py --stage prepare
modal run --profile kieusontung6 experiments/modal/pair_screen_modal.py --stage qwen-nonthinking
modal run --profile kieusontung6 experiments/modal/pair_screen_modal.py --stage qwen-thinking
modal run --profile kieusontung6 experiments/modal/pair_screen_modal.py --stage rescore
```

Set a fresh `PAIR_SCREEN_RUN` for each complete experiment. Stage log files refuse
overwrite. Model cache is separate from run evidence. Existing CPU-only preparation
failures remain preserved; they are not benchmark failures of the model.

Both modes use zero-shot repository math prompts, native chat templates, temperature
0.6, top-p 0.95, top-k 20, seed 42, and 4096 generated tokens. This is a matched screen
at the existing pipeline's token cap, not Qwen's official 32768-token protocol.
Thinking is explicitly switched through the chat template. Incomplete thinking
receives no final answer; it is counted separately from ordinary answer errors.

The repository author-code math grader is pinned to its canonical Git blob SHA;
Windows CRLF is normalized in the persisted remote scorer copy before hash checking.
Raw output, final text, rendered prompt, completion token count and stop metadata
are retained per item. No model-generated code is executed.

The pinned inference runtime lacks the optional ANTLR parser used by the legacy
symbolic grader. `rescore` runs on CPU and installs small grader dependencies into
separate asset directories: ANTLR 4.11.1 for the legacy SymPy grader and
math-verify 0.8.0 with its own dependency set in another directory/process. It
retains both scores, validates fixtures, and reports gold/scorer errors separately.
Neither model inference nor the immutable native runtime is rebuilt for grading.

This screens the public `Qwen/Qwen3-1.7B` checkpoint, not an unverified company
directory or a post-SFT checkpoint. It does not benchmark a Llama teacher, MBPP,
LCB-v6, training gradients, or a full distillation run. A single sampled evaluation
seed cannot support a general efficacy claim.

The Qwen3 technical report tables 18/20 report Llama-3.1-8B-Instruct vs Qwen3-1.7B
non-thinking MATH-500 54.8 vs 73.0 and LCB-v5 10.8 vs 11.6. These published results
motivate checking whether the teacher offers task-specific headroom; model size
does not establish teacher superiority. Exact teacher variant and identical
local evaluation are still required to measure the actual pair's gap.
