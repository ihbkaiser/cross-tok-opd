"""CPU-only contract tests: no model, credentials, downloads or generated user code."""
import argparse
import base64
import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zlib
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contract_eval", ROOT / "scripts/evaluation/contract_eval.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


class ContractTests(unittest.TestCase):
    def test_author_pin_and_profile_prompt_difference(self):
        E.author_helpers()
        item = {"prompt": "What is 1+1?"}
        author = E.messages("gsm8k", item, "author-code", "merge")
        paper = E.messages("gsm8k", item, "paper-spec", "merge")
        self.assertIn("####", author[0]["content"])
        self.assertIn(r"\boxed{}", paper[0]["content"])
        for profile in E.PROFILES:
            mbpp = E.messages("mbpp", {"prompt": "Add", "tests": ["assert add(1,1)==2"]}, profile, "native")
            self.assertEqual(mbpp[1]["content"], "Add\n\nYour code should pass these tests:\nassert add(1,1)==2")

    def test_private_tests_and_functional_metadata(self):
        cases = [{"input": "1", "output": "2", "testtype": "functional"}]
        blob = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()
        row = {"question_id": "q", "question_content": "f", "public_test_cases": "[]",
               "private_test_cases": blob, "metadata": '{"func_name":"f"}'}
        item = E.normalize("live-code-bench-v6", row, "paper-spec")
        self.assertEqual(item["tests"], cases)
        self.assertEqual(item["private_test_count"], 1)
        row["metadata"] = "{}"
        with self.assertRaises(ValueError):
            E.normalize("live-code-bench-v6", row, "paper-spec")
        with self.assertRaises(Exception):
            E.test_list("corrupted hidden tests")

    def test_pickle_globals_rejected_without_execution(self):
        blob = base64.b64encode(zlib.compress(pickle.dumps(Path("x")))).decode()
        with self.assertRaises(ValueError):
            E.test_list(blob)

    def test_generation_parameters_and_seed_are_order_independent(self):
        item = {"id": "abc", "messages": [{"role": "user", "content": "hi"}]}
        a = E.generation_payload("m", item, "mbpp", 42)
        self.assertEqual(a["max_tokens"], 2048)
        self.assertEqual((a["temperature"], a["top_p"], a["n"]), (0.6, 0.95, 1))
        self.assertEqual(a["seed"], E.generation_payload("other-model", item, "mbpp", 42)["seed"])
        self.assertNotEqual(a["seed"], E.generation_payload("m", item, "mbpp", 43)["seed"])

    def test_response_errors_fail_but_empty_completed_text_is_model_output(self):
        for value in ({}, {"choices": []}, {"choices": [{"finish_reason": "stop", "message": {}}]}):
            with self.assertRaises(ValueError):
                E.validate_response(value)
        E.validate_response({"choices": [{"finish_reason": "length", "message": {"content": ""}}]})

    def test_dataset_hash_and_prompt_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            data, out = Path(d)/"data.jsonl", Path(d)/"prepared.json"
            data.write_text(json.dumps({"question": "1+1", "answer": "#### 2"}) + "\n")
            args = argparse.Namespace(format="jsonl", data=str(data), expected_count=1, smoke=True,
                benchmark="gsm8k", profile="paper-spec", system_role="merge", revision="local-fixture",
                split="test", config="main", release=None, out=str(out))
            E.prepare(args)
            prepared = E.load_prepared(out)
            prepared["items"][0]["messages"][0]["content"] = "changed"
            prepared["items_sha256"] = E.digest(E.encoded(prepared["items"]))
            out.write_text(json.dumps(prepared))
            with self.assertRaises(ValueError): E.load_prepared(out)

    def test_duplicate_responses_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"r.jsonl"
            p.write_text('{"id":"a"}\n{"id":"a"}\n')
            with self.assertRaises(ValueError): E.load_records(p, {"a"})

    def test_checkpoint_missing_shard_and_wrong_budget_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"checkpoint"
            p.mkdir()
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                (p/name).write_text("{}")
            (p/"model.safetensors").write_bytes(b"fixture")
            E.checkpoint_identity(p)
            (p/"model.safetensors.index.json").write_text('{"weight_map":{"x":"missing.safetensors"}}')
            with self.assertRaises(ValueError): E.checkpoint_identity(p)
            (p/"model.safetensors.index.json").unlink()
            (p/"run-summary.json").write_text('{"status":"completed","optimizer_updates":5}')
            with self.assertRaises(ValueError): E.checkpoint_identity(p, 312)

    def test_summary_refuses_partial_benchmarks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"m.json"
            p.write_text(json.dumps({"contract":{"checkpoint":{"sha256":"x"},"profile":"paper-spec",
                "benchmark":"gsm8k","seed":42}, "qualification":[], "worker_sha256":"x", "scoring_runner_sha256":"x"}))
            with self.assertRaises(ValueError):
                E.summarize(argparse.Namespace(metrics=[str(p)], out=str(Path(d)/"summary.json")))

    def test_complete_summary_and_mixed_scorer_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for benchmark in E.CAPS:
                for seed in E.SEEDS:
                    p=Path(d)/f"{benchmark}-{seed}.json"
                    p.write_text(json.dumps({"contract":{"checkpoint":{"sha256":"x"},"profile":"author-code",
                        "benchmark":benchmark,"seed":seed,"prepared_sha256":benchmark,"scope":"full"},"qualification":[],
                        "worker_sha256":"w","scoring_runner_sha256":"r","status":"completed","score":0.5}))
                    paths.append(str(p))
            out=Path(d)/"summary.json"
            E.summarize(argparse.Namespace(metrics=paths,out=str(out)))
            self.assertEqual(E.read_json(out)["average"],0.5)
            cell=E.read_json(paths[0]);cell["worker_sha256"]="changed"
            Path(paths[0]).write_text(json.dumps(cell))
            with self.assertRaises(ValueError):
                E.summarize(argparse.Namespace(metrics=paths,out=str(Path(d)/"bad.json")))

    def test_sandbox_has_no_host_home_or_network(self):
        with patch.object(E.shutil, "which", return_value="/usr/bin/bwrap"):
            cmd=E.sandbox_command("/usr/bin/python3")
        self.assertIn("--unshare-all", cmd)
        self.assertIn("--clearenv", cmd)
        self.assertNotIn("/home", cmd)
        with patch.object(E.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError): E.sandbox_command("/usr/bin/python3")

    def test_author_cpu_scoring_fixtures_in_disposable_worker(self):
        # These are hand-written trusted fixtures, never arbitrary downloaded/model code.
        for benchmark in ("gsm8k", "math500", "mbpp", "live-code-bench-v6"):
            for correct in (True, False):
                job={"qualify":True,"profile":"author-code","benchmark":benchmark,"correct":correct}
                result=subprocess.run([sys.executable,str(E.HERE/"contract_worker.py")],
                    input=json.dumps(job), text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(json.loads(result.stdout)["passed"],correct)

    @unittest.skipUnless(E.shutil.which("bwrap"), "bubblewrap not installed")
    def test_real_sandbox_official_lcb_and_memory_cap(self):
        # Existing system Python has NumPy; no installation. Fail clearly if unavailable.
        jobs = [
            ({"tests":[{"input":"2\n3","output":"5"}],"fn_name":"add"}, "def add(a,b): return a+b", True),
            ({"tests":[{"input":"2\n3","output":"5"}],"fn_name":"add"}, "def add(a,b): return 0", False),
            ({"tests":[{"input":"","output":"1"}],"fn_name":None}, "x=bytearray(256*1024*1024);print(1)", False),
        ]
        for item, text, expected in jobs:
            job={"profile":"paper-spec","benchmark":"live-code-bench-v6","item":item,"text":text}
            result=subprocess.run(E.sandbox_command("/usr/bin/python3"),input=E.encoded(job),capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads(result.stdout)["passed"],expected)

    def test_generation_resume_with_real_mock_http_server(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            checkpoint=root/"checkpoint"
            checkpoint.mkdir()
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                (checkpoint/name).write_text("{}")
            (checkpoint/"model.safetensors").write_bytes(b"fixture")
            data=root/"data.jsonl"
            data.write_text('{"question":"1+1","answer":"#### 2"}\n')
            prepared=root/"prepared.json"
            E.prepare(argparse.Namespace(format="jsonl",data=str(data),expected_count=1,smoke=True,benchmark="gsm8k",
                profile="author-code",system_role="merge",revision="fixture",split="test",config="main",release=None,out=str(prepared)))
            requests=[]
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*args): pass
                def send(self,value):
                    self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers()
                    self.wfile.write(json.dumps(value).encode())
                def do_GET(self):
                    self.send({"model_path":str(checkpoint)} if self.path=="/get_server_info" else {"data":[{"id":"fixture-model"}]})
                def do_POST(self):
                    payload=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    requests.append(payload)
                    self.send({"choices":[{"finish_reason":"stop","message":{"content":"#### 2"}}]})
            server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            args=argparse.Namespace(prepared=str(prepared),checkpoint=str(checkpoint),expected_updates=None,
                base_url=f"http://127.0.0.1:{server.server_port}",model="fixture-model",out=str(root/"run"),seed=42,concurrency=2)
            try:
                E.generate(args)
                E.generate(args)
                self.assertEqual(len(requests),1)
                self.assertTrue((root/"run/complete.json").is_file())
                if E.shutil.which("bwrap"):
                    E.score(argparse.Namespace(prepared=str(prepared),run=str(root/"run"),python="/usr/bin/python3",runtime_root=[]))
                    metrics=E.read_json(root/"run/metrics.json")
                    self.assertEqual(metrics["score"],1.0)
                    self.assertEqual(metrics["total"],1)
                args.seed=43
                with self.assertRaises(ValueError): E.generate(args)
            finally:
                server.shutdown();server.server_close();thread.join()


if __name__ == "__main__":
    unittest.main()
