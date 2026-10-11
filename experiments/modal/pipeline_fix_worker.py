"""Qualify production tokenizer/overlap/teacher-ID fixes on cached real assets."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

os.environ["KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT"] = "1"
ROOT = Path("/opt/pipeline_fix")
OUT = Path("/results")


def token_ids(value):
    return list(value["input_ids"]) if hasattr(value, "keys") else list(value)


def tokenizers(arm):
    from kdflow.utils.tokenizer_loader import load_tokenizer
    assets = json.loads(Path("/warmup/ready.json").read_text())
    path = "/phi_teacher/teacher" if arm == "phi" else "/qwen_teacher/teacher"
    return (load_tokenizer(assets["models"][arm], local_files_only=True, trust_remote_code=False),
            load_tokenizer(path, local_files_only=True, trust_remote_code=False), assets["models"][arm], path)


def cpu():
    from tokenizers import Tokenizer
    from kdflow.algorithms._semantic_overlap import semantic_overlap
    from kdflow.datasets.utils import get_tokenizer as data_loader
    from kdflow.utils.utils import get_tokenizer as model_loader
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_semantic_overlap.py",
               "tests/test_tokenizer_loader.py", "tests/test_teacher_forward_chunking.py",
               "tests/test_span_ctkd_metrics.py", "tests/test_exact_trajectory_integration.py",
               "tests/mp_opd/test_grass_chunk.py", "tests/mp_opd/test_grass_chunk_temporal.py",
               "tests/mp_opd/test_grass_span.py", "tests/mp_opd/test_algorithm_audit_regressions.py"]
    subprocess.run(command, cwd=ROOT, check=True)
    records = {}
    for arm in ("phi", "qwen"):
        student, teacher, _, path = tokenizers(arm)
        raw = Tokenizer.from_file(path + "/tokenizer.json")
        fixtures = ["(n", "Let's", "def factorial(n):\n", "café 日本語", "\n\t\r"]
        for loader in (data_loader, model_loader):
            loaded = loader(path)
            for text in fixtures:
                assert loaded.encode(text, add_special_tokens=False) == raw.encode(text, add_special_tokens=False).ids
        messages = [{"role": "user", "content": "Write factorial(n). Let's test it."}]
        text = teacher.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert token_ids(teacher.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)) == raw.encode(text, add_special_tokens=False).ids
        si, ti, record = semantic_overlap(student, teacher)
        assert semantic_overlap(student, teacher) == (si, ti, record)
        assert len(si) == len(set(si)) and len(ti) == len(set(ti))
        for s, t in zip(si[:-1], ti[:-1]):
            assert student.decode([s], clean_up_tokenization_spaces=False) == teacher.decode([t], clean_up_tokenization_spaces=False)
        if arm == "qwen":
            pairs = set(zip(si, ti))
            assert (235248, 220) in pairs  # space, not literal Ġ (241753)
            assert (242750, 198) not in pairs  # literal Ċ is not newline
        records[arm] = {"teacher_loader": teacher.kdflow_tokenizer_provenance,
                        "student_loader": student.kdflow_tokenizer_provenance, "overlap": record,
                        "serialized_fixture_checks": len(fixtures) * 2, "chat_prompt_verified": True}
    (OUT / "cpu.json").write_text(json.dumps(records, indent=2))
    print("PIPELINE_CPU_PASS=" + json.dumps(records), flush=True)


def cache():
    import ray
    sys.path.insert(0, str(ROOT / "experiments/runai"))
    from ray_socket_layout import install_local_socket_layout
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_run_cache.py", "tests/test_runai_contract.py"],
                   cwd=ROOT, check=True)
    persistent = OUT / "ray-cache-check-local-ipc"
    persistent.mkdir(exist_ok=False)
    install_local_socket_layout()
    try:
        ray.init(num_cpus=1, num_gpus=0, object_store_memory=80*1024*1024,
                 _temp_dir=str(persistent), _node_ip_address="127.0.0.1", include_dashboard=False)
        @ray.remote
        def ping(): return 17
        assert ray.get(ping.remote()) == 17
        session = Path(ray._private.worker._global_node.get_session_dir_path())
        assert session.resolve().is_relative_to(persistent.resolve())
        sockets = ray._private.worker._global_node.get_sockets_dir_path()
        assert sockets.startswith("/tmp/rs-")
        result = {"ray": ray.__version__, "ping": 17, "session_path": str(session), "socket_directory": sockets,
                  "persistent_session": str(session.resolve()), "scope": "CPU Ray task startup with persistent session and local IPC; no resume or GPU trainer"}
        (OUT / "cache.json").write_text(json.dumps(result, indent=2))
        print("PIPELINE_CACHE_PASS=" + json.dumps(result), flush=True)
    finally:
        ray.shutdown()


def gpu(arm):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from kdflow.backend.sglang.sglang_engine import SGLangEngineService, EngineConfig
    from kdflow.ray.train.teacher_actor import teacher_input_trajectories
    from kdflow.algorithms.span_ctkd import SpanCrossTokenizerKD, memory_efficient_reverse_kl
    from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer, mp_content_ids
    from kdflow.algorithms._mp_opd_credit import AtomCreditTensors
    from kdflow.algorithms.mp_opd import MetaPartitionedOPD
    from kdflow.algorithms._mp_opd_grass_span import GrassNoiseEstimator
    student, teacher, spath, tpath = tokenizers(arm)
    samples = json.loads((Path("/prior") / (arm + "-baseline.json")).read_text())["samples"][:4]
    jobs = []
    for sample in samples:
        content, _ = mp_content_ids(sample["output_ids"][:96], student)
        text = student.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        prefix = token_ids(teacher.apply_chat_template([{"role": "user", "content": sample["case"]["prompt"]}], tokenize=True, add_generation_prompt=True))
        response = teacher.encode(text, add_special_tokens=False)
        ids = prefix + response + [teacher.eos_token_id]
        mask = np.zeros(len(ids), dtype=bool)
        mask[len(prefix)-1:len(ids)-1] = True
        jobs.append(dict(case=sample["case"]["id"], student_prefix=sample["prompt_ids"], student_content=content,
                         teacher_content=response, ids=ids, mask=mask))
        mb = {"tea_input_ids": torch.tensor([ids]), "tea_attn_mask": torch.ones(1, len(ids), dtype=torch.long)}
        assert teacher_input_trajectories([mb], exact_mode=False) == [ids]
    service = SGLangEngineService(EngineConfig(model_path=tpath, mem_fraction_static=.25,
        context_length=16384, enable_memory_saver=False, enable_weights_cpu_backup=False))
    try:
        service.start(timeout=420)
        hidden = service.generate(prompt=["unused deliberately wrong text"] * len(jobs),
            input_ids=[j["ids"] for j in jobs], loss_masks=[j["mask"] for j in jobs],
            sampling_params={"max_new_tokens": 0}, return_hidden_states=True)
        for j, hs in zip(jobs, hidden):
            assert len(hs) == len(j["teacher_content"]) + 1
            assert np.isfinite(hs).all()
    finally:
        service.shutdown()
    gc.collect(); torch.cuda.empty_cache()
    # Retain only the head, not an independent teacher forward numerical reference.
    model = AutoModelForCausalLM.from_pretrained(tpath, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    head = model.get_output_embeddings()
    del model
    gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(spath, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    algo = SpanCrossTokenizerKD.__new__(SpanCrossTokenizerKD)
    algo.teacher_lm_head, algo.student_tokenizer, algo.teacher_tokenizer = head, student, teacher
    algo.args = SimpleNamespace(kd=SimpleNamespace(span_score_mode="mean_logprob"))
    algo._debug_count = 0; algo._span_debug_count = 0
    algo.student_overlap_token_ids, algo.teacher_overlap_token_ids = algo._find_overlap_tokens()
    atomizer = SimCTAtomizer(student, teacher)
    rows = []
    for job, hs in zip(jobs, hidden):
        ids = job["student_content"]; tids = job["teacher_content"]
        with torch.no_grad():
            output = model(torch.tensor([job["student_prefix"] + ids], device="cuda"), use_cache=False, output_hidden_states=True)
            start = len(job["student_prefix"]) - 1
            logits = output.logits[0, start:start+len(ids)].detach().clone().requires_grad_(True)
            sh = output.hidden_states[-1][0, start:start+len(ids)].detach()
            tl = head(torch.tensor(hs[:-1], device="cuda", dtype=head.weight.dtype)).detach()
        atoms = atomizer.atomize(ids, tids, sample_id=job["case"])
        assert atoms.valid, atoms.failure_reason
        labels = torch.tensor(ids, device="cuda")
        ls = logits.float().log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        lt = tl.float().log_softmax(-1).gather(-1, torch.tensor(tids, device="cuda")[:, None]).squeeze(-1)
        a_s = torch.stack([ls[a.student_start:a.student_end].sum() for a in atoms.atoms])
        a_t = torch.stack([lt[a.teacher_start:a.teacher_end].sum() for a in atoms.atoms])
        weights = torch.tensor([a.student_token_count for a in atoms.atoms], device="cuda")
        base = (a_t - a_s.detach()).detach(); rate = (base / weights).detach()
        credits = AtomCreditTensors(a_t, a_s.detach(), base, weights, rate, -a_s, -ls)
        grass = MetaPartitionedOPD.__new__(MetaPartitionedOPD)
        grass.grass_chunk_source = "run"; grass.grass_chunk_run_length = 2
        grass.grass_geometry = "exact_head"; grass.grass_softcap = float(model.config.final_logit_softcapping)
        grass.grass_head_bias = model.get_output_embeddings().bias is not None
        grass.grass_eps_d = 0.; grass.grass_negative_tol_rel = 1e-6; grass.grass_chunk_shadow = False
        grass.grass_noise = GrassNoiseEstimator(rho=.99, min_adjacent_pairs=8); grass.mode = "grass_chunk"
        loss, metrics = grass._grass_chunk_loss(credits, atoms.atoms, student_logits=logits,
            student_labels=labels, student_hidden=sh, student_ids=ids, teacher_ids=tids)
        gg = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        segments, tea, stu = algo._align_sequences_with_spans(tids, ids)
        assert segments
        sv, tv = algo._build_virtual_vocab_logits(segments, logits, tl, stu, tea)
        simct_loss = memory_efficient_reverse_kl(sv, tv).mean()
        sg = torch.autograd.grad(simct_loss, logits)[0]
        assert all(torch.isfinite(v).all() for v in (loss, simct_loss, gg, sg))
        assert gg.norm() > 0 and sg.norm() > 0
        rows.append({"case": job["case"], "student_tokens": len(ids), "teacher_tokens": len(tids),
            "transport_rows": len(hs), "atoms": len(atoms.atoms), "segments": len(segments),
            "credit_mean": float(base.sum()/weights.sum()), "grass_loss": float(loss),
            "simct_loss": float(simct_loss), "grass_gradient_norm": float(gg.norm()),
            "simct_gradient_norm": float(sg.norm())})
        del output, logits, sh, tl, sv, tv, gg, sg, loss, simct_loss, credits, ls, lt, a_s, a_t
        gc.collect(); torch.cuda.empty_cache()
    result = {"arm": arm, "gpu": torch.cuda.get_device_name(0), "teacher_loader": teacher.kdflow_tokenizer_provenance,
        "overlap": algo.overlap_provenance, "rows": rows,
        "scope": "Actual warmup, production SGLang subprocess/ZMQ/masks, GRASS run2 and SimCT frozen loss gradients. No optimizer, Ray trainer, batch64 trajectory, long4096 or accuracy claim."}
    (OUT / (arm + "-gpu.json")).write_text(json.dumps(result, indent=2))
    print("PIPELINE_GPU_PASS=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "cpu": cpu()
    elif sys.argv[1] in {"cache", "cache-fixed"}: cache()
    else: gpu(sys.argv[1])
