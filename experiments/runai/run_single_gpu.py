import ast
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

mode, limit, output = sys.argv[1:]
assert mode in {"atomic", "fixed", "random", "soft", "gbv", "dpca", "grass", "grass_chunk", "airs", "align"}
limit = int(limit)
assert 0 <= limit <= 312

root = Path(__file__).resolve().parents[2]
run_dir = Path(output).resolve()
resume = os.environ.get("MP_RESUME", "0") == "1"
if resume:
    if not (run_dir / "checkpoints/latest.json").is_file():
        raise ValueError("Resume requested without a complete training checkpoint; refusing SFT restart")
else:
    run_dir.mkdir(parents=True, exist_ok=False)

# Đọc dictionary cấu hình từ runner đã đóng gói, không chạy code Modal.
runner = root / "experiments/modal/mp_opd_phi_gemma_50.py"
tree = ast.parse(runner.read_text())
candidates = [
    n.value
    for n in ast.walk(tree)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "opts" for t in n.targets)
]
assert len(candidates) == 1
call = candidates[0]
assert isinstance(call, ast.Call)
assert isinstance(call.func, ast.Name) and call.func.id == "dict"

opts = {}
for kw in call.keywords:
    assert kw.arg is not None
    if kw.arg.startswith("wandb_"):
        continue
    opts[kw.arg] = ast.literal_eval(kw.value)

shared = Path(os.environ.get("MP_SHARED_ROOT", "/workspace/storage-shared/nlp/tungks/SimCT"))
opts.update(
    num_nodes=1,
    num_gpus_per_node=1,
    student_name_or_path=os.environ.get("MP_STUDENT_PATH", str(
        shared / "runs/qwen-gemma-sft-paper-20260908-045828/checkpoint"
    )),
    teacher_name_or_path=os.environ.get("MP_TEACHER_PATH", "/workspace/storage-shared/models/Qwen2.5-7B-Instruct"),
    train_dataset_path=os.environ.get("MP_DATASET_PATH", str(shared / "data/qwen-author/data/prompts.parquet")),
    num_epochs=2,
    train_batch_size=64,
    micro_train_batch_size=int(os.environ.get('MP_MICRO_TRAIN_BATCH_SIZE', '4')),
    attn_implementation=os.environ.get("MP_ATTN_IMPLEMENTATION", "eager"),
    rollout_num_engines=1,
    rollout_tp_size=1,
    teacher_tp_size=1,
    teacher_dp_size=1,
    mp_opd_mode=mode,
    kd_algorithm=os.environ.get("MP_ALGORITHM", "mp_opd"),
    seed=int(os.environ.get("MP_SEED", "42")),
    lr_scheduler_horizon_steps=312,
    mp_opd_random_seed=int(os.environ.get("MP_PARTITION_SEED", "43")),
    mp_opd_max_span_length=1 if mode == "atomic" else int(os.environ.get("MP_MAX_SPAN_LENGTH", "2")),
    mp_opd_fixed_span_length=int(os.environ.get("MP_FIXED_SPAN_LENGTH", "2")),
    mp_opd_min_span_length=int(os.environ.get("MP_MIN_SPAN_LENGTH", "1")),
    mp_opd_gbv_beta=float(os.environ.get("MP_GBV_BETA", "1.0")),
    mp_opd_gbv_geometry=os.environ.get("MP_GBV_GEOMETRY", "token_count"),
    mp_opd_dpca_clip_ratio_low=float(os.environ.get("MP_DPCA_CLIP_RATIO_LOW", "0.2")),
    mp_opd_dpca_clip_ratio_high=float(os.environ.get("MP_DPCA_CLIP_RATIO_HIGH", "0.28")),
    mp_opd_dpca_clip_ratio_c=float(os.environ.get("MP_DPCA_CLIP_RATIO_C", "10.0")),
    mp_opd_dpca_adv_clamp=float(os.environ.get("MP_DPCA_ADV_CLAMP", "10.0")),
    mp_opd_dpca_agg=os.environ.get("MP_DPCA_AGG", "token-mean"),
    diagnostic_max_updates=limit,
    save_steps=20,
    save_path=str(run_dir / "checkpoint"),
    ckpt_path=str(run_dir / "checkpoints"),
    use_wandb=False,
    load_checkpoint=resume,
    pause_after_updates=int(os.environ.get("MP_PAUSE_AFTER_UPDATES", "0")),
    resume_checkpoint_steps=os.environ.get("MP_CHECKPOINT_STEPS", "40,80,120,156,200,240,280,312"),
    eval_on_checkpoint=os.environ.get("MP_EVAL_ON_CKPT", "0") == "1",
    eval_on_checkpoint_steps=os.environ.get("MP_EVAL_STEPS", "40,80,120,160,200,240,280,312"),
    eval_case_dir=os.environ.get("MP_EVAL_CASE_DIR", ""),
    eval_prepared_dir=os.environ.get("MP_EVAL_PREPARED_DIR", ""),
    eval_concurrency=int(os.environ.get("MP_EVAL_CONCURRENCY", "32")),
    eval_score_python=os.environ.get("MP_EVAL_SCORE_PYTHON", "/usr/bin/python3.12"),
    mp_opd_offload_adam_moments=os.environ.get("MP_OFFLOAD_ADAM_MOMENTS", "0") == "1",
    mp_opd_host_mask=os.environ.get("MP_OPD_HOST_MASK", "0") == "1",
    rollout_deterministic_inference=os.environ.get("MP_ROLLOUT_DETERMINISTIC", "0") == "1",
    rollout_random_seed=int(os.environ.get("MP_ROLLOUT_SEED", "-1")),
    rollout_attention_backend=os.environ.get("MP_ROLLOUT_ATTENTION_BACKEND", ""),
    rollout_disable_radix_cache=os.environ.get("MP_ROLLOUT_DISABLE_RADIX_CACHE", "0") == "1",
)

# GRASS-Chunk reads the chunk from upstream alignment or from a fixed run; the
# dataclass default is the native 'xtoken' source, which fails closed without an
# audited projection. A run that cannot name its chunk source would therefore
# refuse to start for a reason that looks like a bug, so the choice is explicit
# here and echoed to the log rather than left to a default.
# ALIGN reads the chunk from the same source as GRASS-Chunk and applies its own
# projection afterwards, so it inherits that choice rather than duplicating it.
if mode in {"grass_chunk", "align"}:
    opts.update(
        mp_opd_grass_chunk_source=os.environ.get("MP_GRASS_CHUNK_SOURCE", "run"),
        mp_opd_grass_chunk_run_length=int(os.environ.get("MP_GRASS_CHUNK_RUN_LENGTH", "2")),
        mp_opd_grass_chunk_straddle=os.environ.get("MP_GRASS_CHUNK_STRADDLE", "singleton"),
        mp_opd_grass_chunk_shadow=os.environ.get("MP_GRASS_CHUNK_SHADOW", "0") == "1",
    )
    if opts["mp_opd_grass_chunk_source"] == "xtoken":
        projection = os.environ.get("MP_XTOKEN_PROJECTION_PATH")
        expected = os.environ.get("MP_XTOKEN_PROJECTION_SHA256")
        if not projection or not expected:
            raise ValueError(
                "MP_GRASS_CHUNK_SOURCE=xtoken needs MP_XTOKEN_PROJECTION_PATH and "
                "MP_XTOKEN_PROJECTION_SHA256; set MP_GRASS_CHUNK_SOURCE=run for the "
                "fixed-run baseline, which is not an alignment chunk"
            )
        if hashlib.sha256(Path(projection).read_bytes()).hexdigest() != expected:
            raise ValueError("MP_XTOKEN_PROJECTION_SHA256 does not match the file on disk")
        opts.update(xtoken_projection_path=projection, xtoken_projection_sha256=expected)
    print(
        "GRASS_CHUNK_SOURCE="
        + opts["mp_opd_grass_chunk_source"]
        + " run_length="
        + str(opts["mp_opd_grass_chunk_run_length"])
        + " straddle="
        + opts["mp_opd_grass_chunk_straddle"]
        + " shadow="
        + str(opts["mp_opd_grass_chunk_shadow"]),
        flush=True,
    )

if os.environ.get("MP_OFFLOAD_ADAM_MOMENTS", "0") not in {"0", "1"}:
    raise ValueError("MP_OFFLOAD_ADAM_MOMENTS must be 0 or 1")
if os.environ.get("MP_ROLLOUT_DETERMINISTIC", "0") not in {"0", "1"}:
    raise ValueError("MP_ROLLOUT_DETERMINISTIC must be 0 or 1")
if os.environ.get("MP_ROLLOUT_DISABLE_RADIX_CACHE", "0") not in {"0", "1"}:
    raise ValueError("MP_ROLLOUT_DISABLE_RADIX_CACHE must be 0 or 1")
if os.environ.get("MP_EVAL_ON_CKPT", "0") not in {"0", "1"}:
    raise ValueError("MP_EVAL_ON_CKPT must be 0 or 1")
if opts["mp_opd_offload_adam_moments"] and (
    opts["kd_algorithm"] != "mp_opd"
    or mode != "soft"
    or os.environ.get("MP_ALTERNATING", "0") != "1"
):
    raise ValueError("Adam moment offload requires the alternating soft mp_opd production path")

if opts['attn_implementation'] not in {'eager', 'sdpa'}:
    raise ValueError('MP_ATTN_IMPLEMENTATION must be eager or sdpa')
if opts['kd_algorithm'] == 'mp_opd' and opts['attn_implementation'] != 'eager':
    raise ValueError('MP parity qualification requires eager attention')

if mode == "soft":
    energy_path = Path(os.environ['MP_ENERGY_CHECKPOINT'])
    if not energy_path.is_file(): raise ValueError('Missing trained energy checkpoint')
    opts.update(mp_opd_energy_checkpoint=str(energy_path),mp_opd_partition_temperature=1.)
    if os.environ.get('MP_ALTERNATING','0')=='1':
        opts.update(mp_opd_alternating=True,mp_opd_meta_path=os.environ['MP_META_PATH'],
            mp_opd_meta_batch_size=16,
            mp_opd_meta_microbatch_size=int(os.environ.get('MP_META_MICRO_BATCH_SIZE', '4')),
            mp_opd_energy_lr=float(os.environ.get('MP_ENERGY_LR','0.001')),
            mp_opd_energy_every=int(os.environ.get('MP_ENERGY_EVERY','1')))

if os.environ.get('MP_MAX_LEN'):
    # Diagnostic-only: the campaign pins max_len 4096. A short diagnostic may lower it to
    # test whether the padded pair, not the sample count, is what fills the card.
    if not (0 < limit <= 30):
        raise ValueError('MP_MAX_LEN is a diagnostic-only knob (limit 1-30)')
    opts['max_len'] = int(os.environ['MP_MAX_LEN'])

if opts['micro_train_batch_size'] not in (1,2,4,8,16,32,64):
    raise ValueError('Student microbatch must be a positive divisor of B64')
if opts.get('mp_opd_meta_microbatch_size', 4) not in (1,2,4,8,16):
    raise ValueError('Meta microbatch must be a positive divisor of M16')
micro_recipe = (opts['micro_train_batch_size'], opts.get('mp_opd_meta_microbatch_size', 4))
exact_soft_alternating_full = (
    mode == 'soft'
    and limit in (0, opts.get('lr_scheduler_horizon_steps'))
    and micro_recipe == (1, 4)
    and opts.get('kd_algorithm') == 'mp_opd'
    and opts.get('mp_opd_mode') == 'soft'
    and opts.get('mp_opd_alternating') is True
    and opts.get('mp_opd_offload_adam_moments') is True
    and opts.get('attn_implementation') == 'eager'
    and opts.get('train_batch_size') == 64
    and opts.get('max_len') == 4096
    and opts.get('rollout_batch_size') == 64
    and opts.get('generate_max_len') == 4096
    and opts.get('lr_scheduler_horizon_steps') == 312
    and opts.get('exact_token_trajectory') is True
    and opts.get('enforce_max_sequence_length') is True
)
if micro_recipe not in ((4,4),(1,1),(2,4)) and not (0 < limit <= 30) and not exact_soft_alternating_full:
    raise ValueError('Microbatch overrides are restricted to short diagnostics or the exact soft alternating micro1/meta4 full-run contract')
if exact_soft_alternating_full:
    print('ADMISSION_PASS: exact soft alternating micro1/meta4 full-run contract', flush=True)

if opts["kd_algorithm"] not in {"mp_opd", "span_ctkd", "xtoken"}:
    raise ValueError("MP_ALGORITHM must be mp_opd, span_ctkd or xtoken")
if opts["kd_algorithm"] != "mp_opd" and mode != "atomic":
    raise ValueError("use atomic as the neutral launcher slot for non-MP algorithms")
if opts["kd_algorithm"] == "xtoken":
    projection = Path(os.environ["MP_XTOKEN_PROJECTION_PATH"])
    expected = os.environ["MP_XTOKEN_PROJECTION_SHA256"]
    actual = hashlib.sha256(projection.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("X-Token projection checksum mismatch")
    opts.update(xtoken_projection_path=str(projection), xtoken_projection_sha256=expected)
import sys as _sys
from types import SimpleNamespace

if str(root) not in _sys.path:
    _sys.path.insert(0, str(root))
from kdflow.rollout_serving import assert_serving_contract, build_extra_server_args, serving_marker

_serving_namespace = SimpleNamespace(rollout=SimpleNamespace(**{
    key: opts.get(key) for key in (
        "rollout_disable_piecewise_cuda_graph", "rollout_attention_backend",
        "rollout_deterministic_inference", "rollout_disable_radix_cache", "rollout_random_seed",
    )
}))
serving_extra = build_extra_server_args(_serving_namespace)
assert_serving_contract(serving_extra, deterministic=opts["rollout_deterministic_inference"])

if opts["mp_opd_fixed_span_length"] <= 0 or opts["mp_opd_max_span_length"] <= 0:
    raise ValueError("span lengths must be positive")
if mode == "fixed" and opts["mp_opd_fixed_span_length"] > opts["mp_opd_max_span_length"]:
    raise ValueError("fixed length must not exceed MP_MAX_SPAN_LENGTH")

for key in ("student_name_or_path", "teacher_name_or_path", "train_dataset_path"):
    assert Path(opts[key]).exists(), opts[key]

def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

prepared_receipt = None
receipt_path = os.environ.get("MP_PREPARED_RECEIPT")
if receipt_path:
    from experiments.modal.provenance_prepare import load_receipt, prepared_entries, PreparationError
    try:
        prepared_receipt = load_receipt(receipt_path, expected_sha256=os.environ.get("MP_PREPARED_RECEIPT_SHA256") or None, expected_source_commit=os.environ.get("MP_SOURCE_COMMIT") or None, expected_snapshot_id=os.environ.get("MP_SNAPSHOT_ID") or None, verify_files=False)
    except PreparationError as exc:
        raise ValueError("Prepared provenance cache miss; refusing GPU rehash: " + str(exc)) from exc
    print("PREPARED_HASH_REUSE=validated", flush=True)

model_manifest = {}
for role in ("student", "teacher"):
    path = Path(opts[role + "_name_or_path"])
    print(f"Hashing {role} model files for provenance", flush=True)
    if prepared_receipt is not None:
        model_manifest[role] = {k: v["sha256"] for k, v in prepared_entries(prepared_receipt, role, path).items()}
    else:
        model_manifest[role] = {
            str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}
        }
    if not model_manifest[role]:
        raise ValueError("model directory contains no identifiable model files")
versions = {}
for package in ("torch", "transformers", "sglang", "ray"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None

# Operational settings change how resources are laid out, never what the numbers
# are. Record them verbatim in the launch manifest so a diagnostic cannot quietly
# run under a different allocator or CUDA/NCCL configuration than the baseline it
# is compared against.
OPERATIONAL_KEYS = (
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "CUDA_LAUNCH_BLOCKING",
    "NCCL_CUMEM_HOST_ENABLE",
    "NCCL_IB_DISABLE",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_P2P_DISABLE",
    "OMP_NUM_THREADS",
    "RAY_USAGE_STATS_ENABLED",
    "TOKENIZERS_PARALLELISM",
)


def operational_environment(environ=None):
    source = os.environ if environ is None else environ
    return {key: source[key] for key in OPERATIONAL_KEYS if source.get(key)}


def expandable_segments_enabled(recorded):
    joined = "".join(
        str(value) for key, value in recorded.items() if "ALLOC_CONF" in key
    ).replace(" ", "").lower()
    return "expandable_segments:true" in joined

manifest = {
    "qualification_policy": os.environ.get('MP_QUALIFICATION_POLICY','required'),
    "qualification_status": os.environ.get('MP_QUALIFICATION_STATUS','unverified'),
    "options": opts,
    "source_root": str(root),
    "source_commit": os.environ.get("MP_SOURCE_COMMIT"),
    "source_dirty": os.environ.get("MP_SOURCE_DIRTY"),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "variant": mode,
    "energy_sha256": (prepared_entries(prepared_receipt, "energy", energy_path)[energy_path.name]["sha256"] if prepared_receipt is not None else file_hash(energy_path)) if mode == "soft" else None,
    "partition_dp_dtype": "float64" if mode in {"soft", "gbv"} else None,
    "cooperative_stop_at": os.environ.get("MP_TRAIN_STOP_AT"),
    "checkpoint_reserve_seconds": os.environ.get("MP_CHECKPOINT_RESERVE_SECONDS", "300"),
    "models_sha256": model_manifest,
    "dataset_sha256": (prepared_entries(prepared_receipt, "dataset", opts["train_dataset_path"])[Path(opts["train_dataset_path"]).name]["sha256"] if prepared_receipt is not None else file_hash(Path(opts["train_dataset_path"]))),
    "runtime_versions": versions,
    "operational_environment": operational_environment(),
    "contract": {
        "schema": "mp-runai-v2", "algorithm": opts["kd_algorithm"],
        "execution_updates": limit or 312, "scheduler_horizon": 312,
        "sampling_temperature": opts["temperature"],
        "credit_logprob_temperature": 1.0,
        "parity": {"mean_max": 0.1, "p99_max": 0.5} if opts["kd_algorithm"] == "mp_opd" else None,
        "random_rule": "uniform-next-length; not histogram-matched",
        "evaluation_contract": "external pinned eval manifest required",
    },
    "gpu_mapping": "single visible GPU maps SGLang base_gpu_id to zero",
    "serving": {
        "extra_server_args": serving_extra,
        "deterministic": opts["rollout_deterministic_inference"],
        "random_seed": opts["rollout_random_seed"],
        "attention_backend": opts["rollout_attention_backend"] or "auto",
        "disable_radix_cache": opts["rollout_disable_radix_cache"] or None,
        "request_seed_rule": "seeded_sampling(seed, global_step, original_request_index)",
    },
}
if resume:
    previous = json.loads((run_dir / "launch-config.json").read_text())
    def normalized(value):
        value = dict(value)
        options = dict(value["options"])
        for key in ("load_checkpoint", "pause_after_updates"):
            options.pop(key, None)
        value["options"] = options
        # Allocator/CUDA operational settings change memory layout only, never numerics, so
        # they stay out of resume provenance; the resume-attempt manifest records what the
        # resume actually ran under.
        for key in ("cuda_visible_devices", "cooperative_stop_at", "checkpoint_reserve_seconds",
                    "operational_environment"):
            value.pop(key, None)
        return value
    if normalized(previous) != normalized(manifest):
        raise ValueError("Launcher resume provenance differs; original manifest retained")
    import time
    (run_dir / f"resume-attempt-{time.time_ns()}.json").write_text(json.dumps(manifest, indent=2))
else:
    (run_dir / "launch-config.json").write_text(json.dumps(manifest, indent=2))

recorded_operational = manifest["operational_environment"]
if expandable_segments_enabled(recorded_operational):
    print(
        "OPERATIONAL_OVERRIDE expandable_segments=True "
        + json.dumps(recorded_operational, sort_keys=True),
        flush=True,
    )
if os.environ.get("MP_PREFLIGHT_ONLY") == "1":
    print(
        "EFFECTIVE_MP_OPD_OFFLOAD_ADAM_MOMENTS="
        + str(opts["mp_opd_offload_adam_moments"]).lower(),
        flush=True,
    )
    print(
        "EFFECTIVE_MP_ROLLOUT_DETERMINISTIC="
        + str(opts["rollout_deterministic_inference"]).lower(),
        flush=True,
    )
    print("EFFECTIVE_SERVING_ARGS=" + serving_marker(serving_extra), flush=True)
    print(
        "EFFECTIVE_OPERATIONAL_ENV=" + json.dumps(operational_environment(), sort_keys=True),
        flush=True,
    )
    print(f"PREFLIGHT_READY={run_dir / 'launch-config.json'}")
    raise SystemExit(0)

import ray
ray.init(
    address="local",
    num_gpus=1,
    num_cpus=8,
    include_dashboard=False,
    _temp_dir=os.environ["MP_RAY_TMP"],
    runtime_env={"env_vars": {key: os.environ[key] for key in
        ("MP_PARITY_CAPTURE_DIR", "MP_SOURCE_COMMIT", "KDFLOW_ROLLOUT_PORT_BASE",
         "KDFLOW_ROUTER_PORT_BASE", "KDFLOW_ROUTER_PROMETHEUS_PORT") if key in os.environ}},
)

try:
    import kdflow.cli.train_kd_on_policy as cli

    original = cli.create_placement_group

    def single_gpu_placement(num_gpus):
        assert num_gpus == 1
        pg, indices, gpu_ids = original(num_gpus)
        assert len(gpu_ids) == 1
        print(
            f"SINGLE_GPU_MAPPING: visible={os.environ['CUDA_VISIBLE_DEVICES']}, "
            f"ray_ids={gpu_ids}, sglang_local_id=0",
            flush=True,
        )
        return pg, indices, [0]

    cli.create_placement_group = single_gpu_placement

    sys.argv = ["train_kd_on_policy"]
    for key, value in opts.items():
        sys.argv.extend(["--" + key, str(value)])

    # Diagnostic backend with Gemma attention softcapping.
    args = cli.init_args()
    # TensorBoard is off by default and no runai entry point turns it on, so a run
    # that wants the curves has to ask for them here. Set the nested field after
    # init_args for the same reason as max_len below: the flag is nested under
    # args.log, so there is no flat --log.use_tensorboard on the command line.
    if os.environ.get("MP_TENSORBOARD", "0") == "1":
        args.log.use_tensorboard = True
        tb_dir = os.environ.get("MP_TENSORBOARD_DIR", "").strip()
        if tb_dir:
            args.log.tensorboard_log_dir = tb_dir
        print(
            "TENSORBOARD_ON dir=%s"
            % (args.log.tensorboard_log_dir or os.path.join(args.train.save_path, "tensorboard")),
            flush=True,
        )
    if os.environ.get('MP_MAX_LEN'):
        # init_args raises max_len back to prompt_max_len + generate_max_len, so the flag
        # alone would leave a manifest that disagrees with what actually ran. Bind the
        # effective value once the arguments exist, before any rollout reads it.
        args.data.max_len = int(os.environ['MP_MAX_LEN'])
    print('EFFECTIVE_DATA_MAX_LEN=%d' % int(args.data.max_len), flush=True)
    args.model.attn_implementation = opts['attn_implementation']
    print(f"TRAIN_ATTN_OVERRIDE={opts['attn_implementation']}", flush=True)
    cli.train(args)

    summary = json.loads(
        (run_dir / "checkpoint/run-summary.json").read_text()
    )
    expected = limit or 312
    if summary["status"] == "stopped" and summary.get("stop_reason") in {"deadline_checkpoint_reserve", "checkpoint_pause"}:
        updates = summary["optimizer_updates"]
        checkpoint = run_dir / "checkpoint" / f"step{updates}"
        assert updates >= 1 and checkpoint.is_dir(), summary
        print(f"RUN_PARTIAL_SAVED: {mode}, {updates}/{expected} updates; checkpoint={checkpoint}", flush=True)
    else:
        assert summary["status"] == "completed", summary
        assert summary["optimizer_updates"] == expected, summary
        print(f"RUN_VERIFIED: {mode}, {expected} updates", flush=True)
finally:
    ray.shutdown()