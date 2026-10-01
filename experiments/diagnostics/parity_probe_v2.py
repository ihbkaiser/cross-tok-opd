"""Corrected parity probe for the pod (GPU 0).

Why this version exists: an earlier Modal probe (P1) reported a divergence starting at token
2048, but that was an artefact of the probe itself -- the trainer reference was computed in
2048-token chunks with no position offset, so every token past 2048 was scored as if the chunk
were a fresh sequence. The lesson is baked in here:

  1. the trainer reference is ONE forward pass over the whole sequence (no chunking);
  2. a SELF-CHECK runs first: on a short sequence, engine and trainer must agree to <0.05,
     otherwise the measurement itself is broken and the script says so instead of reporting a
     fake divergence;
  3. it measures the training's real reading path (generated-token logprobs) under BOTH a single
     request and the concurrent batch width training uses, because the engine only ever saw
     single requests in the earlier probe.

Marker lines: PROBE_SELFCHECK, PROBE_MODE=<solo|batch|concurrency>, PROBE_BAND=..., PROBE_RECEIPT

Run on the pod:
  CUDA_VISIBLE_DEVICES=0 bash /workspace/storage-shared/nlp/tungks/simct-b200-portable-539d29f/experiments/runai/python-b200-host.sh \
    /usr/bin/python3.12 <this file> --case <case dir> --out <case dir>/logs/parity_probe
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

PORT = 31700
PHRASE = ("The quick brown fox jumps over the lazy dog. "
          "Numbers, punctuation and spacing matter for tokenisation; ")


def start_engine(model_dir, mem_fraction, context, extra, log_path):
    argv = ["python", "-m", "sglang.launch_server", "--model-path", model_dir,
            "--served-model-name", "probe", "--host", "127.0.0.1", "--port", str(PORT),
            "--tp-size", "1", "--mem-fraction-static", str(mem_fraction),
            "--context-length", str(context)] + shlex.split(extra)
    log = open(log_path, "wb")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            return proc, False
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2):
                return proc, True
        except Exception:
            time.sleep(2)
    return proc, False


def post(body, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d/generate" % PORT,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def bands(deltas, width):
    out = []
    for lo in range(0, len(deltas), width):
        seg = deltas[lo:lo + width]
        if seg:
            out.append([lo, lo + len(seg), round(sum(seg) / len(seg), 6), round(max(seg), 4)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--concurrency", default="1,8,64")
    ap.add_argument("--gen", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--band", type=int, default=512)
    ap.add_argument("--mem-fraction", type=float, default=0.25)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--extra-flags", default="--disable-piecewise-cuda-graph")
    ap.add_argument("--trainer-fp32-samples", type=int, default=1)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = Path(args.case)
    model_dir = args.model
    if not model_dir:
        model_dir = json.loads((case / "campaign.json").read_text())["student"]
    result = {"schema": "simct-parity-probe-v2", "case": str(case), "model_dir": model_dir,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "selfcheck": {}, "modes": {}}

    def write():
        try:
            (out_dir / "parity-probe.json").write_text(json.dumps(result, indent=2, default=str))
        except Exception as exc:
            print("PROBE_RECEIPT_FAILED=" + repr(exc), flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids_all = tok(PHRASE * 400, return_tensors=None)["input_ids"]
    while len(ids_all) < args.gen:
        ids_all = (ids_all + ids_all)
    prompt = ids_all[:args.prompt_tokens]
    result["prompt_tokens"] = len(prompt)

    def trainer_logprobs(seq, dtype):
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, attn_implementation="eager").to("cuda").eval()
        with torch.no_grad():
            part = torch.tensor([seq], device="cuda")
            logits = model(part).logits.float()[0]
            lp = logits.log_softmax(-1)
            vals = [float(lp[pos - 1, seq[pos]]) for pos in range(1, len(seq))]
        del model, logits, lp, part
        torch.cuda.empty_cache()
        return vals

    engine = None
    try:
        engine, ok = start_engine(model_dir, args.mem_fraction, args.context,
                                  args.extra_flags, out_dir / "server.log")
        result["engine_ready"] = ok
        if not ok:
            raise RuntimeError("engine failed to start, see server.log")

        # ---- SELF-CHECK: short sequence, engine prompt logprobs vs single-pass trainer ----
        short = ids_all[:256]
        ref_short = trainer_logprobs(short, torch.bfloat16)
        payload = post({"input_ids": short, "return_logprob": True, "logprob_start_len": 0,
                        "top_logprobs_num": 0,
                        "sampling_params": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 1}})
        ent = (payload.get("meta_info") or {}).get("input_token_logprobs") or []
        eng_short = [float(e[0]) for e in ent if isinstance(e, list) and e[0] is not None]
        n = min(len(eng_short), len(ref_short))
        d = [abs(eng_short[i] - ref_short[i]) for i in range(n)]
        result["selfcheck"] = {"n": n, "mean": sum(d) / n if n else None,
                               "max": max(d) if d else None}
        ok_self = bool(d) and max(d) < 0.05
        print("PROBE_SELFCHECK=" + json.dumps(result["selfcheck"]), flush=True)
        if not ok_self:
            print("PROBE_SELFCHECK_FAILED=measurement unreliable, stopping", flush=True)
            result["status"] = "selfcheck_failed"
            return result

        # ---- measure: same prompts, solo then at training's batch width ----
        def one_request(prompt_ids):
            p = post({"input_ids": prompt_ids, "return_logprob": True, "logprob_start_len": 0,
                      "top_logprobs_num": 0,
                      "sampling_params": {"temperature": 0.6, "top_p": 0.95,
                                          "max_new_tokens": args.gen}})
            meta = p.get("meta_info") or {}
            lp = [float(e[0]) for e in (meta.get("output_token_logprobs") or [])
                  if isinstance(e, list) and e[0] is not None]
            gid = [int(t) for t in (p.get("output_ids") or [])]
            return lp, gid

        def compare(tag, prompt_ids, lps, gids, fp32_budget):
            seq = prompt_ids + gids[:len(lps)]
            if len(seq) < 2:
                return None
            ref = trainer_logprobs(seq, torch.bfloat16)
            m = min(len(lps), len(ref) - args.prompt_tokens)
            if m <= 0:
                return None
            delta = [abs(lps[i] - ref[args.prompt_tokens + i]) for i in range(m)]
            entry = {"n": m, "mean": sum(delta) / m, "max": max(delta),
                     "count_gt_1": sum(1 for v in delta if v > 1.0),
                     "onset_gt_1": next((i for i, v in enumerate(delta) if v > 1.0), None),
                     "bands": bands(delta, args.band)}
            if fp32_budget > 0:
                ref32 = trainer_logprobs(seq, torch.float32)
                d32 = [abs(lps[i] - ref32[args.prompt_tokens + i]) for i in range(m)]
                db = [abs(ref[i] - ref32[i]) for i in range(args.prompt_tokens, args.prompt_tokens + m)]
                entry["engine_vs_fp32"] = {"mean": sum(d32) / m, "max": max(d32)}
                entry["bf16trainer_vs_fp32"] = {"mean": sum(db) / m, "max": max(db)}
            return entry

        solo_lp, solo_gid = one_request(prompt)
        solo = compare("solo", prompt, solo_lp, solo_gid, args.trainer_fp32_samples)
        result["modes"]["solo"] = solo
        print("PROBE_MODE=solo " + json.dumps(solo, default=str)[:900], flush=True)
        write()

        for width in [int(x) for x in args.concurrency.split(",") if x.strip()]:
            t0 = time.time()
            with cf.ThreadPoolExecutor(max_workers=width) as pool:
                futs = [pool.submit(one_request, prompt) for _ in range(width)]
                got = [f.result() for f in futs]
            per = []
            for i, (lp, gid) in enumerate(got):
                entry = compare("batch%d" % width, prompt, lp, gid, 0)
                if entry:
                    per.append(entry)
            if per:
                agg = {"width": width, "samples": len(per),
                       "mean": sum(e["mean"] for e in per) / len(per),
                       "max": max(e["max"] for e in per),
                       "worst_count_gt_1": max(e["count_gt_1"] for e in per),
                       "seconds": round(time.time() - t0, 1)}
                result["modes"]["batch%d" % width] = agg
                print("PROBE_MODE=batch%d %s" % (width, json.dumps(agg)), flush=True)
                print("PROBE_BAND=batch%d %s" % (width, json.dumps(per[0]["bands"])), flush=True)
                write()
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:400])
        print("PROBE_STATUS=failed " + repr(exc)[:300], flush=True)
    finally:
        try:
            if engine is not None and engine.poll() is None:
                engine.terminate()
                engine.wait(timeout=30)
        except Exception:
            try:
                engine.kill()
            except Exception:
                pass
        result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write()
        print("PROBE_RECEIPT=" + str(out_dir / "parity-probe.json"), flush=True)
    return result


if __name__ == "__main__":
    main()
