"""One scoring job inside bubblewrap. Do not invoke directly on model outputs."""
import contextlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import sys

from contract_eval import HERE, author_helpers, digest, read_json


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_lcb():
    path = HERE / "vendor/lcb_testing_util.py"
    provenance = read_json(HERE / "vendor/provenance.json")
    if digest(path.read_bytes()) != provenance["sha256"]:
        raise ValueError("Pinned official LCB scorer hash mismatch")
    spec = importlib.util.spec_from_file_location("official_lcb_testing_util", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def math_score(h, benchmark, profile, item, text):
    if profile == "author-code":
        if benchmark == "gsm8k":
            pred = h["extract_number_from_answer"](text)
            gold = h["extract_number_from_answer"](item["gold"])
            return pred is not None and gold is not None and abs(pred - gold) < 1e-6
        return bool(h["is_math_equivalent"](h["extract_boxed_answer"](text), item["gold"]))
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    gold = item["gold"].split("####")[-1].strip() if benchmark == "gsm8k" else item["gold"]
    target = parse("$" + gold + "$", extraction_config=[LatexExtractionConfig()])
    if not target:
        raise ValueError("Gold math answer unparseable; do not score it as a model error")
    prediction = parse(text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    return bool(prediction and verify(target, prediction))


def fixture(benchmark, correct):
    if benchmark in {"gsm8k", "math500"}:
        return {"gold": "2"}, r"\boxed{2}" if correct else r"\boxed{3}"
    if benchmark == "mbpp":
        return {"tests": ["assert add(2, 3) == 5", "assert add(1, 1) == 2"], "setup": ""}, (
            "def add(a,b): return a+b" if correct else "def add(a,b): return a-b")
    return {"tests": [{"input": "2 3\n", "output": "5\n"}], "fn_name": None}, (
        "a,b=map(int,input().split());print(a+b)" if correct else "print(0)")


def main():
    job = json.load(sys.stdin)
    benchmark, profile = job["benchmark"], job["profile"]
    if profile not in {"author-code", "paper-spec"}:
        raise ValueError("Unknown profile")
    h = author_helpers()
    versions = {name: package_version(name) for name in ("math-verify", "sympy", "numpy", "antlr4-python3-runtime")}
    official = load_lcb() if profile == "paper-spec" and benchmark == "live-code-bench-v6" else None
    if profile == "paper-spec" and benchmark in {"gsm8k", "math500"}:
        import math_verify  # mandatory; never fall back to the author string checker
    # Per-process limits are additional to OS namespace isolation, not a substitute.
    if profile == "paper-spec" and benchmark in {"mbpp", "live-code-bench-v6"}:
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if job.get("qualify"):
        item, text = fixture(benchmark, job["correct"])
    else:
        item, text = job["item"], job["text"]
    text = h["strip_thinking_content"](text)[0]
    # Never mix program stdout with the trusted JSON protocol.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        if benchmark in {"gsm8k", "math500"}:
            passed = math_score(h, benchmark, profile, item, text)
        else:
            code = h["_extract_code_block"](text)
            if benchmark == "mbpp":
                if profile == "author-code":
                    passed = h["_run_mbpp_tests"](code, item["tests"], item["setup"], timeout=10.0)
                else:
                    # Paper reconstruction: same author assertion semantics, 10s per test.
                    # The paper does not identify an MBPP harness revision; do not claim otherwise.
                    passed = all(h["_run_mbpp_tests"](code, [test], item["setup"], timeout=10.0)
                                 for test in item["tests"])
            elif profile == "author-code":
                num_passed, total = h["_run_lcb_tests"](code, item["tests"], timeout=10.0)
                passed = total > 0 and num_passed == total
            else:
                sample = {"input_output": json.dumps({"inputs": [t["input"] for t in item["tests"]],
                          "outputs": [t["output"] for t in item["tests"]], "fn_name": item["fn_name"]})}
                results, _ = official.run_test(sample, test=code, timeout=10)
                passed = len(results) == len(item["tests"]) and all(bool(x == True) for x in results)
    result = {"passed": bool(passed)}
    if job.get("qualify"):
        if bool(passed) != job["correct"]:
            raise ValueError("Scorer qualification failed")
        result.update(versions=versions, qualified=True,
                      lcb_revision=read_json(HERE / "vendor/provenance.json")["commit"] if official else None)
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
