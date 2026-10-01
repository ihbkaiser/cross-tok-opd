"""P1d + P1e in one run, on profile tungkieu6868.

P1d: read the engine's logprobs the way TRAINING does -- generate 256 tokens after a 2048-token
prompt, then compare the engine's output_token_logprobs against the trainer over exactly the
[2048, 2304) window that P1 localised. P1 prompt logprobs were already tested; this tests the
generation path in the same window.

P1e: does any engine knob clean the window? Two candidate configs, each measured with the same
prompt-mode probe as P1 so the numbers are directly comparable with 0.82491 / 22.6138.

  uvx --from modal modal run modal_p1de_sweep_20260930.py::sweep
"""
from __future__ import annotations

import json
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

IMAGE_REF = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
ASSET_VOLUME = "simct-qwen7b-gemma2-assets-20260916"
RUN_VOLUME = "simct-qwen7b-gemma2-runs-20260916-main"
APP_NAME = "simct-p1de-sweep-20260930"
PORT = 31000
OUT_DIR = "/runs/p1de_sweep"
PHRASE = ("The quick brown fox jumps over the lazy dog. "
          "Numbers, punctuation and spacing matter for tokenisation; ")
CONFIGS = [("baseline", "--disable-piecewise-cuda-graph"),
           ("kv-fp32", "--disable-piecewise-cuda-graph --kv-cache-dtype fp32"),
           ("noradix", "--disable-piecewise-cuda-graph --disable-radix-cache")]

app = modal.App(APP_NAME)
image = modal.Image.from_registry(IMAGE_REF).entrypoint([])
assets = modal.Volume.from_name(ASSET_VOLUME, create_if_missing=False)
runs = modal.Volume.from_name(RUN_VOLUME, create_if_missing=False)


def _find_model(root: Path, subdir: str) -> Path:
    base = root / subdir
    for cand in [base] + sorted(p for p in base.rglob("*") if p.is_dir()):
        if (cand / "config.json").is_file():
            return cand
    raise RuntimeError("no model dir under " + str(base))


@app.function(image=image, gpu="B200", cpu=8, memory=49152, timeout=3000, retries=0,
              max_containers=1, single_use_containers=True,
              volumes={"/assets": assets, "/runs": runs})
def sweep(prompt_len: int = 2048, gen: int = 256, long_len: int = 4096,
          only: str = "") -> dict:
    """only: comma-separated config names to run; empty means all. One config keeps the spend
    inside a nearly exhausted budget."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    result: dict = {"schema": "simct-p1de-sweep-v1", "prompt_len": prompt_len, "gen": gen,
                    "long_len": long_len, "gpu": torch.cuda.get_device_name(0), "configs": {}}

    def write() -> None:
        try:
            (out_dir / ("sweep-" + stamp + ".json")).write_text(
                json.dumps(result, indent=2, default=str))
        except Exception as exc:
            print("SWEEP_RECEIPT_FAILED=" + repr(exc), flush=True)
        try:
            runs.commit()
        except Exception:
            pass

    model_dir = _find_model(Path("/assets"), "student")
    result["model_dir"] = model_dir
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(PHRASE * 400, return_tensors=None)["input_ids"]
    while len(ids) < long_len:
        ids = (ids + ids)[:long_len]
    ids = ids[:long_len]
    long_ids = ids[:long_len]
    short_prompt = ids[:prompt_len]

    def trainer_over(token_ids, first_scored):
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
        outs = []
        with torch.no_grad():
            part = torch.tensor([token_ids], device="cuda")
            logits = model(part).logits.float()[0]
            lp = logits.log_softmax(-1)
            for pos in range(first_scored, len(token_ids)):
                outs.append(float(lp[pos - 1, token_ids[pos]]))
            del logits, lp, part
        del model
        torch.cuda.empty_cache()
        return outs

    def start_server(extra):
        argv = ["python", "-m", "sglang.launch_server", "--model-path", model_dir,
                "--served-model-name", "probe", "--host", "127.0.0.1", "--port", str(PORT),
                "--tp-size", "1", "--mem-fraction-static", "0.25", "--context-length", "8192"] \
               + shlex.split(extra)
        proc = subprocess.Popen(argv, stdout=open(out_dir / "server.log", "ab"),
                                stderr=subprocess.STDOUT)
        deadline = time.time() + 900
        while time.time() < deadline:
            if proc.poll() is not None:
                return proc, False, "exited"
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2):
                    return proc, True, "ready"
            except Exception:
                time.sleep(2)
        return proc, False, "timeout"

    def stop(proc):
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        time.sleep(5)

    def post(body):
        req = urllib.request.Request("http://127.0.0.1:%d/generate" % PORT, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.loads(resp.read().decode())

    try:
        wanted = [s.strip() for s in only.split(",") if s.strip()]
        selected = [(n, e) for n, e in CONFIGS if not wanted or n in wanted]
        result["selected"] = [n for n, _ in selected]
        for name, extra in selected:
            entry: dict = {"extra": extra}
            result["configs"][name] = entry
            proc = None
            try:
                proc, ok, why = start_server(extra)
                entry["server"] = why
                if not ok:
                    continue
                # P1e: prompt logprobs over the whole long sequence (comparable with P1)
                payload = post({"input_ids": long_ids, "return_logprob": True, "logprob_start_len": 0,
                                "top_logprobs_num": 0,
                                "sampling_params": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 1}})
                entries = (payload.get("meta_info") or {}).get("input_token_logprobs") or []
                eng = [float(e[0]) for e in entries if isinstance(e, list) and e[0] is not None]
                tr = trainer_over(long_ids, 1)
                n = min(len(eng), len(tr))
                band = [abs(eng[i] - tr[i]) for i in range(2048, min(2304, n))]
                entry["prompt_band_2048_2304"] = {"n": len(band),
                                                  "mean": sum(band) / len(band) if band else None,
                                                  "max": max(band) if band else None}
                # P1d: training's reading path -- generate after a 2048-token prompt
                gen_payload = post({"input_ids": short_prompt, "return_logprob": True,
                                    "logprob_start_len": 0, "top_logprobs_num": 0,
                                    "sampling_params": {"temperature": 0.6, "top_p": 0.95,
                                                        "max_new_tokens": gen}})
                out_entries = (gen_payload.get("meta_info") or {}).get("output_token_logprobs") or []
                out_lp = [float(e[0]) for e in out_entries if isinstance(e, list) and e[0] is not None]
                gen_ids = []
                try:
                    gen_ids = [int(t) for t in gen_payload.get("output_ids") or []]
                except Exception:
                    gen_ids = []
                entry["generated_tokens"] = len(out_lp)
                if out_lp and len(gen_ids) >= len(out_lp):
                    full = short_prompt + gen_ids[:len(out_lp)]
                    tr_gen = trainer_over(full, prompt_len)
                    m = min(len(out_lp), len(tr_gen))
                    d = [abs(out_lp[i] - tr_gen[i]) for i in range(m)]
                    entry["output_path_2048_2304"] = {
                        "n": m, "mean": sum(d) / m if m else None, "max": max(d) if d else None,
                        "onset_gt_1": next((i for i, v in enumerate(d) if v > 1.0), None)}
            except BaseException as exc:
                entry.update(error_type=type(exc).__name__, error=str(exc)[:300])
            finally:
                stop(proc)
                write()
            print("SWEEP_CONFIG=" + json.dumps({name: result["configs"][name]}, default=str)[:600],
                  flush=True)
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:300])
    finally:
        write()
    print("SWEEP_STATUS=" + str(result.get("status")), flush=True)
    print("SWEEP_RECEIPT=" + str(out_dir / ("sweep-" + stamp + ".json")), flush=True)
    return {"status": result.get("status"), "configs": result.get("configs")}
