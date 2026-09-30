# SimCT evaluation contracts

This runner reuses the released SimCT prompts, extraction helpers and author-code
graders. It does not use the company evalhub scorer. No command starts a server,
installs dependencies, monitors checkpoints, kills jobs or uploads results.

## Source and two explicit profiles

`evaluation.py` is byte-identical to upstream `sunjie279/SimCT-` at
`cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e`; SHA256
`21378cfb1aa1d2f3ddab684a1bcb671fd588919c76fd410fac424bd062db2839`.
The runner verifies this hash and compiles a whitelist of unchanged helpers.
Upstream source: https://github.com/sunjie279/SimCT-/blob/cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e/scripts/evaluation/evaluation.py
Paper: https://arxiv.org/html/2605.07711v1 (Appendix C).

| Contract | `author-code` | `paper-spec` |
|---|---|---|
| GSM8K prompt | Author #### prompt | Author boxed math prompt |
| Math checker | Unchanged author numeric/string/SymPy helpers | Mandatory Math-Verify; no fallback |
| MBPP prompt/extraction | Author zero-shot, tests in prompt | Same |
| MBPP execution | Author helper; 10s for all tests together | Author assertion helper, separate 10s per test, 128 MiB address-space cap |
| LCB prompt/extraction | Author stdin/stdout prompt | Same, to preserve the released prompt |
| LCB grader | Author stdin/stdout helper | Pinned official LCB `run_test`, including functional tasks |
| LCB timeout | Author 10s/test, total capped at 120s | Official 10s/test; 128 MiB address-space cap |

Both profiles explicitly use T=0.6, top-p=0.95, n=1, zero-shot, and token caps
4096 (GSM8K/MATH-500/LCB), 2048 (MBPP), matching the author's shell recipe and
paper. Evaluation seeds 42..46 are our convention, not disclosed author seeds.
Each request uses a deterministic hash of repeat seed and item ID, independent
of model and scheduling. Backend nondeterminism is still possible.

`author-code` preserves author prompt/grader semantics, not every operational
quirk: strict API failures, explicit seeds, sandbox, dataset validation and
provenance are added. `paper-spec` is an explicit reconstruction, NOT an assertion
that the exact hidden author evaluation harness has been recovered:

- Paper does not identify an MBPP harness commit. The per-test wrapper above is
  derived from the author code, not falsely labelled an official MBPP executable.
- LCB source is pinned in `vendor/provenance.json`, MIT license included. Its
  revision is ours; the paper does not identify its evaluator revision.
- The released LCB prompt requests stdin/stdout even for functional tasks. We
  retain that prompt and disclose the mismatch; no silent prompt replacement.
- MBPP variant, dataset revisions, exact seeds, math-verify version, extraction
  configuration and meaning of 128 MB are incompletely specified in the paper.
  We implement the cap as 128 MiB RLIMIT_AS (includes Python/library overhead).
- Math-Verify uses LatexExtractionConfig then ExprExtractionConfig, with gold
  boxed as LaTeX. Gold parse failures fail the cell; no numeric fallback.

Never mix profiles in one comparison table. Report these differences alongside
scores; do not call either profile an exact reproduction of Table 1.

## 1. CPU scorer preflight before using any GPU

Use an existing evaluation interpreter. Required for generation/JSON preparation:
Python stdlib only. `hf-disk`/`hf-cache` require existing `datasets`. Scoring needs
Linux bubblewrap with unprivileged namespaces; paper math additionally needs
math-verify and its compatible ANTLR runtime; official LCB needs NumPy. Author
MATH uses optional SymPy as in upstream. Exact package versions are recorded in
every scored cell; summary refuses version drift between repeats.

```bash
/usr/bin/python3.12 scripts/evaluation/contract_eval.py preflight \
  --profile author-code --python /usr/bin/python3.12

/usr/bin/python3.12 scripts/evaluation/contract_eval.py preflight \
  --profile paper-spec --python /usr/bin/python3.12
```

This runs only synthetic correct/wrong CPU fixtures in a sandbox. A missing
dependency or disabled namespaces returns nonzero; there is no unsandboxed
fallback. Do not install into the training venv while jobs are running. If a
separate existing interpreter uses `/opt`, pass narrowly scoped `--runtime-root`
for its Python and venv directories; never mount home, shared data, or secrets.
`/usr`, system libraries and library-loader metadata are read-only; home and
company shared storage are absent, network/PID namespaces isolated, environment
cleared. Process limits are not themselves a sandbox. Sandbox availability and
qualification must be checked on the company node, not inferred from local WSL.

## 2. Prepare local company data (offline)

If the company LCB export lacks metadata, `download_lcb_v6.py` downloads the
six canonical raw shards at pinned HF revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505` through an explicit company proxy.
It requires stdlib Python only, not datasets' remote-code loader. Raw downloads
total about 4.18 GiB; retaining raw shards and merged JSONL needs about 8.36 GiB
plus reserve. It verifies each LFS SHA256, validates 1055 unique IDs and functional
metadata, preserves every raw field, and writes `manifest.json`. Run on company
storage only; this is not a request to download data onto the personal machine.

```bash
/usr/bin/python3.12 scripts/evaluation/download_lcb_v6.py \
  --proxy http://10.30.154.118:80 \
  --output /workspace/storage-shared/nlp/tungks/simct-eval-data/lcb-v6-0fe84c3
```

Re-running resumes `.part` downloads and re-verifies completed files. A normal
failure releases `.download.lock`; after a hard kill inspect the process before
removing that specific stale lock. Corrupt completed files are not overwritten.
Only `LCB_V6_READY` indicates successful validation; a downloaded shard is not
completion. This does not solve code-scoring isolation or authorize GPU work.

Create output outside the training checkout, e.g. a dedicated directory under
`/workspace/storage-shared/nlp/tungks/simct-eval/`. Use an isolated eval checkout
or source export while training is active; do not run updater in its checkout.

```bash
python3 scripts/evaluation/contract_eval.py prepare \
  --profile author-code --benchmark mbpp \
  --format hf-cache --data google-research-datasets/mbpp --config full \
  --revision ACTUAL_CACHED_HF_COMMIT --split test --expected-count 500 \
  --system-role merge --out /absolute/eval-dir/mbpp-author-code.json
```

The example contains a required revision placeholder. Set `HF_HOME` to the
existing company cache before invoking `hf-cache`; all HF loading is offline.
Do not guess a revision to make a command pass. `--format hf-disk --data /path`
uses `load_from_disk`; `--format jsonl --data /path/file.jsonl` uses stdlib.
Revision is user-supplied provenance, not cryptographically proven by the local
file. The prepared artifact hashes all actual normalized rows and prompts.

`--system-role merge` implements the author's no-system-role path for Gemma,
merging the system instruction into the user message; use consistently across
compared Gemma checkpoints. `native` keeps two roles. No automatic fallback.

Required JSONL schema:

- GSM8K: `question`, `answer` (including original #### gold); optional `id`.
- MATH-500: `problem`, `answer`; optional `unique_id`.
- MBPP: `task_id`, `text`, `test_list`; optional `test_setup_code`.
- LCB: `question_id`, `question_content`, `public_test_cases`, `private_test_cases`,
  `metadata` (including `func_name` for functional tasks), and explicit
  `--release release_v6`. Preserves private test encoding; restricted unpickler
  disallows globals. Corrupt/missing hidden tests fail instead of being dropped.

Expected complete reference sizes are GSM8K test 1319, MATH-500 test 500,
MBPP full test 500, LCB release_v6 1055. Verify content/IDs against the actual
snapshot, not just counts. MBPP full IDs 11..510 are the official full test split;
the paper does not disambiguate full versus sanitized. Source:
https://github.com/google-research/google-research/tree/master/mbpp
LCB release reference: https://github.com/LiveCodeBench/LiveCodeBench#dataset-versions

The full profile enforces these counts and MBPP IDs 11..510. Use `--smoke` only
for explicitly small diagnostics; smoke artifacts cannot produce a full summary.

The company LCB JSONL `id/prompt/ground_truth` cannot be assumed lossless. If
original IDs, function metadata or private tests are absent, do not fabricate
them. Obtain the original cached snapshot or a verified lossless export. This
is a data blocker, not a reason to switch to evalhub's partial-test mean score.

## 3. Generate using a separately started, authorized SGLang server

Only after the user assigns GPU evaluation and training/checkpoints are complete.
The server must already serve the exact immutable checkpoint. The runner checks
`/get_server_info` model_path and `/v1/models` identity before/after each cell.
It hashes checkpoint weights/tokenizer; this is I/O-heavy and should follow
training completion. The path check alone cannot prove the server was restarted
after a file replacement; use immutable checkpoint paths and preserve server
launch provenance. Keep the source and runtime versions fixed for all models.

```bash
for seed in 42 43 44 45 46; do
  python3 scripts/evaluation/contract_eval.py generate \
    --prepared /absolute/eval-dir/mbpp-author-code.json \
    --checkpoint /absolute/final-run/checkpoint --expected-updates 312 \
    --model /absolute/final-run/checkpoint --base-url http://127.0.0.1:30000 \
    --seed "$seed" --concurrency 8 \
    --out "/absolute/eval-dir/atomic/author-code/mbpp/seed-$seed" || break
done
```

`--expected-updates 312` requires summary completed/312 and sibling run exit0.
Omit it for an independently verified SFT/base checkpoint. Output must be unique
per model/profile/benchmark/seed. Existing responses resume only under identical
dataset/checkpoint/request contracts; duplicate IDs and truncated JSONL fail.
After an abrupt kill, inspect results before removing only that cell's stale
`.generating` lock. No automatic stale-lock deletion. No key is sent; only local
HTTP endpoints are supported. No inference-error-to-empty-answer conversion.

## 4. Score saved responses on CPU, independently of GPU serving

```bash
python3 scripts/evaluation/contract_eval.py score \
  --prepared /absolute/eval-dir/mbpp-author-code.json \
  --run /absolute/eval-dir/atomic/author-code/mbpp/seed-42 \
  --python /usr/bin/python3.12
```

The scorer verifies the completion receipt, all IDs and request hashes, then
qualifies the selected grader with correct/wrong fixtures. Errors in the sandbox
or dependencies abort the cell. `metrics.json` is written only when every item
is scored, with boolean pass per problem and response/worker/version provenance.
This intentionally does not report partial-test credit as pass@1. Original raw
responses, usage and finish_reason are retained in `responses.jsonl`.

## 5. Aggregate only complete 4-benchmark x 5-seed results

```bash
python3 scripts/evaluation/contract_eval.py summarize \
  --metrics /absolute/eval-dir/atomic/author-code/*/seed-*/metrics.json \
  --out /absolute/eval-dir/atomic-author-code-summary.json
```

Requires exactly 20 cells for one checkpoint/profile. Reports each benchmark
mean, both sample/population std, and unweighted average of four means. These
are evaluation-repeat variations, not independent training seeds. No partial
overall scores, no benchmark mixing, no W&B uploads.

## Verification boundary

Run CPU regression tests with existing stdlib Python:
`python3 -m unittest discover -s tests -p test_eval_contract.py -v`.
Tests use synthetic prompts, a local mock HTTP server and trusted code fixtures.
They do not reproduce model quality, remote data identity, or GPU serving.
Paper math qualification remains unavailable until math-verify exists in the
selected scoring environment. No dependency installation is performed here.

Verified locally in WSL on 2026-09-09: 14 unittest cases passed, including the
prepare/mock-HTTP-generation/resume/sandbox-score flow, functional LCB and a
256-MiB allocation rejected under the paper cap. All four author-code CPU
preflights and paper-spec MBPP/LCB preflights passed. An additional infinite-loop
LCB fixture was rejected by the 10-second timeout. Paper math preflight returned
not_ready for missing math_verify, as intended. Company runtime and real data
remain unverified; no GPU evaluation was run.
