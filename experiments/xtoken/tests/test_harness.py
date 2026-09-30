"""Offline stdlib checks: never import torch, download models, or launch a GPU job."""
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))
import run_logged
import xtoken


class RootTests(unittest.TestCase):
    def test_reject_relative(self):
        with self.assertRaises(ValueError):
            xtoken.validate_root("relative")

    def test_reject_broad_root(self):
        for root in ("/", "/tmp", str(Path.home()), str(xtoken.SIMCT), str(xtoken.SIMCT.parent)):
            with self.subTest(root=root), self.assertRaises(ValueError):
                xtoken.validate_root(root)

    def test_reject_inside_worktree(self):
        with self.assertRaises(ValueError):
            xtoken.validate_root(str(EXPERIMENT / "runtime"))

    def test_accept_external_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(xtoken.validate_root(tmp), Path(tmp).resolve())

    def test_reject_managed_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "NeMo-RL").symlink_to(EXPERIMENT, target_is_directory=True)
            with self.assertRaises(ValueError):
                xtoken.validate_root(tmp)


class CheckoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        (self.repo / "uv.lock").write_text("locked fixture\n")
        self.pin = dict(xtoken.PIN, uv_lock_sha256=hashlib.sha256(b"locked fixture\n").hexdigest())
        self.dirty = ""
        self.head = self.pin["commit"]
        self.origin = self.pin["repository"]
        self.top = str(self.repo)
        self.subs = "\n".join(f" {sha} {name} (pinned)" for name, sha in self.pin["submodules"].items())

    def git(self, repo, *args):
        if args == ("rev-parse", "--show-toplevel"):
            return self.top
        if args == ("remote", "get-url", "origin"):
            return self.origin
        if args == ("rev-parse", "HEAD"):
            return self.head
        if args[0] == "status":
            return self.dirty
        raise AssertionError(args)

    def verify(self):
        with patch.object(xtoken, "PIN", self.pin), patch.object(xtoken, "git", self.git), \
                patch.object(xtoken.subprocess, "check_output", return_value=self.subs):
            return xtoken.verify_checkout(self.repo)

    def test_exact_pin_passes(self):
        self.assertEqual(self.verify()["head"], self.head)

    def test_wrong_head_rejected(self):
        self.head = "0" * 40
        with self.assertRaises(ValueError):
            self.verify()

    def test_wrong_origin_rejected(self):
        self.origin = "https://example.invalid/other.git"
        with self.assertRaises(ValueError):
            self.verify()

    def test_parent_git_checkout_rejected(self):
        self.top = str(self.repo.parent)
        with self.assertRaises(ValueError):
            self.verify()

    def test_dirty_rejected(self):
        self.dirty = " M pyproject.toml"
        with self.assertRaises(ValueError):
            self.verify()

    def test_wrong_lock_rejected(self):
        (self.repo / "uv.lock").write_text("changed")
        with self.assertRaises(ValueError):
            self.verify()

    def test_uninitialized_submodule_rejected(self):
        self.subs = "-" + self.subs[1:]
        with self.assertRaises(ValueError):
            self.verify()

    def test_changed_submodule_rejected(self):
        self.subs = "+" + self.subs[1:]
        with self.assertRaises(ValueError):
            self.verify()

    def test_missing_submodule_rejected(self):
        self.subs = "\n".join(self.subs.splitlines()[:-1])
        with self.assertRaises(ValueError):
            self.verify()

    def test_missing_repo_rejected(self):
        with self.assertRaises(ValueError):
            xtoken.verify_checkout(self.repo / "missing")


class CliTests(unittest.TestCase):
    def test_check_does_not_launch_or_require_uv(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(xtoken, "verify_checkout", return_value={"head": "fixture"}), \
                patch.object(xtoken, "run_logged") as runner, patch.object(xtoken, "resolve_uv") as uv:
            self.assertEqual(xtoken.main(["--root", tmp, "check"]), 0)
            runner.assert_not_called()
            uv.assert_not_called()

    def test_existing_prepare_is_verification_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "NeMo-RL").mkdir()
            with patch.object(xtoken, "verify_checkout", return_value={}), \
                    patch.object(xtoken, "run_logged") as runner:
                self.assertEqual(xtoken.main(["--root", tmp, "prepare"]), 0)
                runner.assert_not_called()

    def test_gate_failure_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(xtoken, "verify_checkout", side_effect=ValueError("bad pin")), \
                patch.object(xtoken, "run_logged") as runner:
            self.assertEqual(xtoken.main(["--root", tmp, "install-base"]), 1)
            runner.assert_not_called()

    def test_uv_mismatch_rejected(self):
        with patch.object(xtoken.shutil, "which", return_value="/tmp/uv"), \
                patch.object(xtoken.subprocess, "check_output", return_value="uv 0.0.0\n"):
            with self.assertRaises(ValueError):
                xtoken.resolve_uv("uv")

    def test_uv_exact_accepted(self):
        with patch.object(xtoken.shutil, "which", return_value="/tmp/uv"), \
                patch.object(xtoken.subprocess, "check_output", return_value=f"uv {xtoken.PIN['uv_version']} (test)\n"):
            self.assertEqual(xtoken.resolve_uv("uv"), "/tmp/uv")

    def test_auth_environment_not_forwarded(self):
        injected = {"HF_TOKEN": "dummy", "WANDB_API_KEY": "dummy", "KAGGLE_KEY": "dummy",
                    "MODAL_TOKEN_SECRET": "dummy", "SERVICE_PASSWORD": "dummy", "PATH": "/bin"}
        with patch.dict(os.environ, injected, clear=True):
            self.assertEqual(run_logged.workload_environment(), {"PATH": "/bin"})

    def test_workload_env_and_nonzero_result_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result"
            out.mkdir()
            with patch.dict(os.environ, {"VIRTUAL_ENV": "/wrong/venv", "UV_PROJECT_ENVIRONMENT": "/wrong/env",
                                         "UV_NO_SYNC": "true", "PYTHONPATH": "/wrong/imports"}), \
                    patch.object(xtoken, "verify_checkout", return_value={"head": "fixture"}), \
                    patch.object(xtoken, "resolve_uv", return_value="/fixture/uv"), \
                    patch.object(xtoken, "run_logged", return_value=(7, out)) as runner:
                self.assertEqual(xtoken.main(["--root", tmp, "install-base"]), 7)
                runner.assert_called_once()
                env = runner.call_args.kwargs["env"]
                for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_NO_SYNC", "PYTHONPATH"):
                    self.assertNotIn(key, env)
                self.assertEqual(env["UV_CACHE_DIR"], str(Path(tmp) / "uv-cache"))
                self.assertEqual(env["HF_HOME"], str(Path(tmp) / "hf-cache"))
                self.assertEqual(env["XTOKEN_UV_BIN"], "/fixture/uv")
                self.assertTrue((out / "source-before.json").exists())
                self.assertTrue(json.loads((out / "source-after.json").read_text())["source_verified"])

    def test_changed_source_after_success_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result"
            out.mkdir()
            with patch.object(xtoken, "verify_checkout", side_effect=[{}, ValueError("dirty")]), \
                    patch.object(xtoken, "resolve_uv", return_value="/fixture/uv"), \
                    patch.object(xtoken, "run_logged", return_value=(0, out)):
                self.assertEqual(xtoken.main(["--root", tmp, "test-base"]), 1)
                self.assertFalse(json.loads((out / "source-after.json").read_text())["source_verified"])


@unittest.skipUnless(os.name == "posix", "POSIX process groups required")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = run_logged.workload_environment()
        # Prevent the offline tests from touching a GPU, even on a CUDA host.
        self.probe = patch.object(run_logged.subprocess, "run", side_effect=FileNotFoundError())
        self.probe.start()
        self.addCleanup(self.probe.stop)

    def run_command(self, code, timeout=3):
        return run_logged.run_logged([sys.executable, "-c", code], cwd=self.root,
                                     root=self.root / "artifacts", name="fixture",
                                     timeout=timeout, env=self.env)

    def test_unique_output_and_success(self):
        first, out1 = self.run_command("print('first')")
        second, out2 = self.run_command("print('second')")
        self.assertEqual((first, second), (0, 0))
        self.assertNotEqual(out1, out2)
        self.assertEqual((out1 / "stdout.log").read_text().strip(), "first")
        self.assertEqual(json.loads((out1 / "result.json").read_text())["attempt"], 1)

    def test_failure_not_retried(self):
        rc, out = self.run_command("from pathlib import Path; Path('once').open('a').write('x'); raise SystemExit(7)")
        self.assertEqual(rc, 7)
        self.assertEqual((self.root / "once").read_text(), "x")
        self.assertEqual(json.loads((out / "result.json").read_text())["exit_code"], 7)

    def test_missing_telemetry_does_not_kill_workload(self):
        rc, out = self.run_command("import time; time.sleep(0.05)")
        result = json.loads((out / "result.json").read_text())
        self.assertEqual(rc, 0)
        self.assertEqual(result["telemetry_samples"], 0)
        self.assertIn("FileNotFoundError", result["telemetry_errors"])

    def test_disconnected_console_preserves_success_failure_and_timeout(self):
        class ClosedReader:
            def write(self, text):
                raise BrokenPipeError()
            def flush(self):
                raise BrokenPipeError()

        for code, timeout, expected in [("print('finished')", 3, 0),
                ("raise SystemExit(7)", 3, 7), ("import time; time.sleep(30)", 0.1, 124)]:
            with self.subTest(expected=expected), patch.object(sys, 'stdout', ClosedReader()):
                rc, out = self.run_command(code, timeout=timeout)
            result = json.loads((out / 'result.json').read_text())
            self.assertEqual(rc, expected)
            self.assertEqual(result['exit_code'], expected)
            self.assertTrue(result['console_disconnected'])

    def test_real_pipe_reader_disconnect_does_not_abort_or_exit_120(self):
        code = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(EXPERIMENT / 'scripts')!r}); "
            "from run_logged import run_logged, workload_environment; "
            f"root=Path({str(self.root)!r}); "
            "rc,_=run_logged([sys.executable,'-c','import time; time.sleep(0.3); print(123)'],"
            "cwd=root,root=root/'pipe',name='fixture',timeout=3,env=workload_environment()); sys.exit(rc)"
        )
        driver = subprocess.Popen([sys.executable, '-c', code],
                                  env=dict(self.env, PATH='/nonexistent'),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertTrue(driver.stdout.readline().startswith(b'EVIDENCE_DIR='))
            driver.stdout.close()
            self.assertEqual(driver.wait(timeout=8), 0)
            self.assertNotIn(b'BrokenPipeError', driver.stderr.read())
            result_path = next((self.root/'pipe').glob('*/result.json'))
            self.assertEqual(json.loads(result_path.read_text())['exit_code'], 0)
            self.assertTrue(json.loads(result_path.read_text())['console_disconnected'])
            self.assertEqual((result_path.parent/'stdout.log').read_text().strip(), '123')
        finally:
            if driver.poll() is None:
                driver.kill()
                driver.wait()
            driver.stderr.close()

    def test_console_other_io_errors_are_not_suppressed(self):
        class DiskFailure:
            def write(self, text):
                raise OSError(28, 'fixture disk full')
        with self.assertRaises(OSError):
            run_logged.BestEffortOutput(DiskFailure()).write('x')

    def test_timeout_has_terminal_evidence(self):
        rc, out = self.run_command("import time; time.sleep(30)", timeout=0.1)
        result = json.loads((out / "result.json").read_text())
        self.assertEqual(rc, 124)
        self.assertTrue(result["timed_out"])
        self.assertLess(result["elapsed_seconds"], 7)

    def test_spawn_failure_recorded(self):
        rc, out = run_logged.run_logged(["/nonexistent/xtoken-fixture"], cwd=self.root,
                                       root=self.root / "artifacts", name="fixture",
                                       timeout=1, env=self.env)
        self.assertEqual(rc, 127)
        self.assertEqual(json.loads((out / "result.json").read_text())["error_type"], "FileNotFoundError")

    def test_leftover_descendant_terminated(self):
        code = ("import subprocess, sys; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                "Path('descendant.pid').write_text(str(p.pid))")
        rc, _ = self.run_command(code)
        self.assertEqual(rc, 0)
        pid = int((self.root / "descendant.pid").read_text())
        state_file = Path(f"/proc/{pid}/stat")
        # An orphan may briefly remain as a zombie awaiting the host's reaper.
        for _ in range(20):
            try:
                state = state_file.read_text().split()[2]
            except (FileNotFoundError, ProcessLookupError):
                # The reaper may remove /proc/<pid>/stat during the read itself.
                break
            if state == "Z":
                break
            time.sleep(0.01)
        else:
            os.kill(pid, signal.SIGKILL)
            self.fail("descendant still running after parent exit")

    def test_invalid_name_rejected(self):
        with self.assertRaises(ValueError):
            run_logged.run_logged([sys.executable], cwd=self.root, root=self.root,
                                  name="../escape", timeout=1, env=self.env)

    def test_ctrl_c_cleans_child_and_records_interruption(self):
        # The child driver has no nvidia-smi on PATH, so this remains an offline test.
        code = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(EXPERIMENT / 'scripts')!r}); "
            "from run_logged import run_logged, workload_environment; "
            f"root=Path({str(self.root)!r}); "
            "rc,_=run_logged([sys.executable,'-c',"
            "\"import os,time; from pathlib import Path; Path('ready.pid').write_text(str(os.getpid())); time.sleep(30)\"],"
            "cwd=root, root=root/'interrupt', name='fixture', timeout=30, env=workload_environment()); "
            "sys.exit(rc)"
        )
        driver = subprocess.Popen([sys.executable, "-c", code], cwd=self.root,
                                  env=dict(self.env, PATH="/nonexistent"),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while not (self.root / "ready.pid").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((self.root / "ready.pid").exists())
            driver.send_signal(signal.SIGINT)
            driver.communicate(timeout=10)
            self.assertEqual(driver.returncode, 130)
            reports = list((self.root / "interrupt").glob("*/result.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertTrue(report["interrupted"])
            child_pid = int((self.root / "ready.pid").read_text())
            self.assertFalse(Path(f"/proc/{child_pid}").exists())
        finally:
            if driver.poll() is None:
                driver.kill()
                driver.communicate()


class RealGitTests(unittest.TestCase):
    def test_prepare_script_with_local_origin_and_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "origin"
            source.mkdir()
            runtime = Path(tmp) / "runtime"
            def git(*args):
                return subprocess.check_output(
                    ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-C", str(source), *args],
                    text=True, stderr=subprocess.STDOUT).strip()
            git("init")
            (source / "uv.lock").write_text("fixture lock\n")
            git("add", "uv.lock")
            git("commit", "-m", "fixture")
            pin = dict(xtoken.PIN, repository=str(source), commit=git("rev-parse", "HEAD"),
                       submodules={}, uv_lock_sha256=hashlib.sha256(b"fixture lock\n").hexdigest())
            actual_run = subprocess.run
            def no_gpu(command, *args, **kwargs):
                if command[0] == "nvidia-smi":
                    raise FileNotFoundError()
                return actual_run(command, *args, **kwargs)
            with patch.object(xtoken, "PIN", pin), patch.object(subprocess, "run", side_effect=no_gpu):
                self.assertEqual(xtoken.main(["--root", str(runtime), "prepare"]), 0)
                self.assertEqual(xtoken.main(["--root", str(runtime), "prepare"]), 0)
                self.assertEqual(len(list((runtime / "artifacts").glob("*/result.json"))), 1)
                self.assertEqual(xtoken.verify_checkout(runtime / "NeMo-RL")["head"], pin["commit"])

    def test_real_git_pin_pass_and_dirty_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            env = run_logged.workload_environment()
            def run(*args):
                return subprocess.check_output(
                    ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
                    stderr=subprocess.STDOUT, env=env, text=True).strip()
            run("init")
            run("remote", "add", "origin", xtoken.PIN["repository"])
            (repo / "uv.lock").write_text("fixture lock\n")
            run("add", "uv.lock")
            run("commit", "-m", "fixture")
            pin = dict(xtoken.PIN, commit=run("rev-parse", "HEAD"), submodules={},
                       uv_lock_sha256=hashlib.sha256(b"fixture lock\n").hexdigest())
            with patch.object(xtoken, "PIN", pin):
                self.assertEqual(xtoken.verify_checkout(repo)["head"], pin["commit"])
                (repo / "uv.lock").write_text("changed")
                with self.assertRaises(ValueError):
                    xtoken.verify_checkout(repo)


class PackagingTests(unittest.TestCase):
    def test_python_scripts_parse_without_importing_torch(self):
        for path in (EXPERIMENT / "scripts").glob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text())

    def test_no_machine_specific_paths_in_scripts(self):
        for path in (EXPERIMENT / "scripts").iterdir():
            if path.is_file():
                text = path.read_text()
                self.assertNotIn("/workspace/xtoken-native", text)
                self.assertNotIn("/home/tung", text)

    def test_no_quantized_or_model_training_claim(self):
        evidence = json.loads((EXPERIMENT / "reports" / "historical-evidence.json").read_text())
        self.assertFalse(evidence["pretrained_model_training_validated"])
        self.assertFalse(evidence["automodel_worker_validated"])
        self.assertFalse(evidence["opd_validated"])
        self.assertEqual(evidence["pip_check_exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
