"""Read-only CPU inventory; never imports Torch, launches GPU, or executes tests."""
import argparse
import importlib.metadata
import json
from pathlib import Path


def inspect_model(model):
    result = {"id": model["id"]}
    checkpoint = model.get("checkpoint")
    if "launch_log" in model:
        launch = Path(model["launch_log"])
        if not launch.is_file():
            return dict(result, state="launch_log_missing")
        runs = [line.removeprefix("RUN_DIR=").strip()
                for line in launch.read_text().splitlines() if line.startswith("RUN_DIR=")]
        if len(runs) != 1 or not Path(runs[0]).is_absolute():
            return dict(result, state="ambiguous_run_dir")
        run = Path(runs[0])
        checkpoint = str(run / "checkpoint")
        exit_file = Path(str(run) + ".exitcode")
        result["exitcode"] = exit_file.read_text().strip() if exit_file.is_file() else None
    path = Path(checkpoint)
    result["checkpoint"] = str(path)
    result["exists"] = path.is_dir()
    result["config_present"] = (path / "config.json").is_file()
    result["weight_file_count"] = len(list(path.glob("*.safetensors"))) + len(list(path.glob("pytorch_model*.bin")))
    summary_path = path / "run-summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
            result["summary"] = {key: summary.get(key) for key in
                                 ("status", "optimizer_updates", "stop_reason")}
        except (ValueError, OSError) as error:
            result["summary_error"] = type(error).__name__
    result["state"] = "inventory_only_not_validated"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("eval_protocol.json"))
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    packages = {}
    for name in ("torch", "sglang", "transformers", "datasets", "math-verify",
                 "antlr4-python3-runtime", "aiohttp", "livecodebench"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    data = {}
    for benchmark in protocol["benchmarks"]:
        path = args.data_root / benchmark["name"] if args.data_root else None
        data[benchmark["name"]] = {"path": str(path) if path else None,
                                    "exists": path.is_dir() if path else None}
    print(json.dumps({"status": "inventory_only_not_eval_ready", "packages": packages,
                      "datasets": data,
                      "models": [inspect_model(model) for model in protocol["model_candidates"]]},
                     indent=2))


if __name__ == "__main__":
    main()
