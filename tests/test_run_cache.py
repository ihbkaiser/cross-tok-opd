"""Exercise shell runner cache persistence without models, imports or a GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix", "Bash node runner")
class RunCacheTests(unittest.TestCase):
    def test_persistent_cache_and_ray_session_root(self):
        with tempfile.TemporaryDirectory(prefix="run-cache-") as raw:
            root = Path(raw)
            code = root / "code"
            scripts = code / "experiments/runai"
            scripts.mkdir(parents=True)
            shutil.copy2(ROOT / "experiments/runai/run_single_gpu.sh", scripts / "run_single_gpu.sh")
            (scripts / "python-b200-host.sh").write_text(
                '#!/bin/bash\nprintf \'{"cache":"%s","ray":"%s"}\\n\' "$XDG_CACHE_HOME" "$MP_RAY_TMP"\n')
            subprocess.run(["git", "init", "-q", str(code)], check=True)
            subprocess.run(["git", "add", "experiments/runai/run_single_gpu.sh", "experiments/runai/python-b200-host.sh"], cwd=code, check=True)
            subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                            "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], cwd=code, check=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("MP_", "JM_"))}
            env.update(MP_TASK_CACHE_ROOT=str(root / "shared"), MP_RUN_ROOT=str(root / "runs"))
            result = subprocess.run(["bash", str(scripts / "run_single_gpu.sh"), "3", "grass_chunk", "2"],
                                    cwd=code, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            values = json.loads(next((root / "runs").glob("*.log")).read_text())
            cache, session_root = Path(values["cache"]), Path(values["ray"])
            self.assertEqual(cache.parent, root / "shared")
            self.assertEqual(session_root, cache / "ray")
            self.assertTrue(cache.is_dir())


if __name__ == "__main__": unittest.main()
