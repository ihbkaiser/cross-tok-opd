"""Prepare immutable assets and run the actual node launcher on Modal B200."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/results/source")
EXPECTED = os.environ["GUARDED_CANARY_SOURCE_COMMIT"]
FIXTURES = Path("/results/fixture-prompts.parquet")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def prepare():
    if not ROOT.exists():
        subprocess.run(["git", "clone", "-b", "vdt/ops/b200-portable",
                        "/work/source.bundle", str(ROOT)], check=True)
    assert git("rev-parse", "HEAD") == EXPECTED
    assert not git("status", "--porcelain", "--untracked-files=no")
    teachers = {}
    for arm in ("qwen", "phi"):
        base = Path("/"+arm+"_assets")
        candidates = [p.parent for p in base.rglob("config.json")
                      if any(p.parent.glob("*.safetensors")) and "teacher" in str(p.parent)]
        if len(candidates) != 1:
            raise ValueError("Expected one cached "+arm+" teacher: "+str(candidates))
        path = candidates[0]
        config = json.loads((path/"config.json").read_text())
        expected_type = "qwen2" if arm == "qwen" else "phi3"
        assert config["model_type"] == expected_type, (arm, config["model_type"])
        teachers[arm] = str(path)
    sys.path.insert(0, str(ROOT/"experiments/modal"))
    from warmup_debug_prepare import main as import_warmup
    import_warmup()
    from datasets import Dataset
    data = json.loads((ROOT/"experiments/modal/warmup_debug_inputs.json").read_text())
    cases = data["cases"]
    rows = [{"messages": [{"role": "user", "content": cases[i % len(cases)]["prompt"]}],
             "fixture_case": cases[i % len(cases)]["id"], "source": "diagnostic_fixture"}
            for i in range(128)]
    Dataset.from_list(rows).to_parquet(str(FIXTURES))
    ready = json.loads(Path("/warmup/ready.json").read_text())
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT/"experiments/modal/vendor")+":"+str(ROOT),
               KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT="1", CUDA_VISIBLE_DEVICES="")
    tests = ["tests/mp_opd/test_guarded_launch.py", "tests/test_runai_contract.py",
             "tests/test_run_cache.py", "tests/test_tokenizer_loader.py"]
    subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, env=env, check=True)
    teacher_hashes = {arm: {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob("*"))
                           if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".txt", ".bin"}}
                      for arm, path in teachers.items()}
    receipt = {"source_commit": EXPECTED, "teachers": teachers, "teacher_sha256": teacher_hashes,
               "warmup": ready, "dataset_sha256": sha(FIXTURES), "fixture_rows": 128,
               "scope": "Diagnostic fixture prompts with real company SFT weights; not training dataset reproduction"}
    Path("/results/prepared.json").write_text(json.dumps(receipt, indent=2))
    print("GUARDED_ASSETS_JSON="+json.dumps({k:v for k,v in receipt.items() if k not in {"warmup", "teacher_sha256"}}), flush=True)


def canary(stage):
    arm, mode = stage.split("-", 1)
    assert mode in {"align_new", "trust_p"} and arm in {"qwen", "phi"}
    receipt = json.loads(Path("/results/prepared.json").read_text())
    assert receipt["source_commit"] == EXPECTED and git("rev-parse", "HEAD") == EXPECTED
    assert not git("status", "--porcelain", "--untracked-files=no")
    assert sha(FIXTURES) == receipt["dataset_sha256"]
    # The image already provides host libraries. This canary runs the same shell
    # wrapper with its host-library directory contract, without rebuilding it.
    runtime = Path("/tmp/guarded-runtime")
    (runtime/"runtime-host-libs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(MP_STUDENT_PATH=receipt["warmup"]["models"][arm],
               MP_TEACHER_PATH=receipt["teachers"][arm], MP_DATASET_PATH=str(FIXTURES),
               MP_RUNTIME_DIR=str(runtime), MP_RUN_DIR="/results/"+stage+"-run",
               MP_RUN_ROOT="/results", MP_TASK_CACHE_ROOT="/results/task-cache",
               MP_GENERATE_MAX_LEN="128", MP_MAX_LEN="4096", MP_TRUST_SCOPE="batch",
               MP_MICRO_TRAIN_BATCH_SIZE="4", MP_GRASS_CHUNK_SOURCE="xtoken",
               MP_GRASS_CHUNK_RUN_LENGTH="2", MP_EVAL_ON_CKPT="0",
               MP_TENSORBOARD="1", MP_TENSORBOARD_DIR="/results/"+stage+"-tensorboard",
               MP_QUALIFICATION_POLICY="implementation-canary", MP_QUALIFICATION_STATUS="in-progress")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "sglang", "ray")}
    print("GUARDED_CANARY_ENV_JSON="+json.dumps({"arm":arm,"mode":mode,"source_commit":EXPECTED,
        "versions":versions,"train_batch_size":64,"micro_train_batch_size":4,
        "generate_max_len":128,"max_len":4096,"expected_updates":2,
        "align_chunk_source":"xtoken" if mode == "align_new" else None,
        "scope":"Production launcher/FSDP/Ray/SGLang, real SFT, short responses; not accuracy"}), flush=True)
    command = ["bash", str(ROOT/"experiments/runai/run_single_gpu.sh"), "0", mode, "2"]
    process = subprocess.run(command, cwd=ROOT, env=env)
    if process.returncode:
        run_log = Path(env["MP_RUN_DIR"]+".log")
        if run_log.exists(): print(run_log.read_text()[-18000:], flush=True)
        raise RuntimeError("Production launcher exit "+str(process.returncode))
    output = Path(env["MP_RUN_DIR"])
    config = json.loads((output/"launch-config.json").read_text())
    summary = json.loads((output/"checkpoint/run-summary.json").read_text())
    opts = config["options"]
    assert opts["mp_opd_mode"] == mode and opts["train_batch_size"] == 64 and opts["micro_train_batch_size"] == 4
    assert opts["generate_max_len"] == 128 and opts["max_len"] == 4096
    assert summary["status"] == "completed" and summary["optimizer_updates"] == 2, summary
    print("GUARDED_TRAIN_SUMMARY_JSON="+json.dumps(summary), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare": prepare()
    else: canary(sys.argv[1])
