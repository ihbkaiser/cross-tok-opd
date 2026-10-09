"""Data layer for the mathplus eval extension (AIME24/25/26, GPQA-Diamond, AMC23).

Standalone beside the main 4-bench contract: this module never modifies
``queue_data``/``contract_eval``/``evaluation.py``, so the running eval queues
keep validating. New files only.
"""

import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

PROFILE = "company-internal-mathplus-v1"
SEEDS = (42, 43, 44)

# Bench -> gen cap (new tokens). Campaign server context is 8192 total.
CAPS = {"aime24": 4096, "aime25": 4096, "aime26": 4096,
        "amc23": 2048, "gpqa-diamond": 2048}
BENCHES = tuple(CAPS)

# (repo, revision, remote path, sha256-of-file). Revisions pinned 2026-10-09.
# GPQA sha is the dataset HEAD commit, not the file hash: the file hash is
# verified at download time and recorded in the prepared source record.
PINNED = {
    "aime24": ("Maxwell-Jia/AIME_2024", "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
               "aime_2024_problems.parquet", None),
    "aime25": ("math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b",
               "test.jsonl", None),
    "aime26": ("math-ai/aime26", "79037aebdb6580008fb960d17cb21fd3099083e3",
               "aime2026.jsonl", None),
    "amc23": ("zwhe99/amc23", "f9810c0439cd3c670ec885d328a2f06a87f3694a",
              "data/test-00000-of-00001.parquet", None),
    "gpqa-diamond": ("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6",
                     "gpqa_diamond.csv", None),
}
# Expected row counts (test split / diamond config).
COUNTS = {"aime24": 30, "aime25": 30, "aime26": 30,
          "amc23": 40, "gpqa-diamond": 198}

MATH_BENCHES = ("aime24", "aime25", "aime26", "amc23")

QWEN_MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
GPQA_SYSTEM_PROMPT = (
    "You are a helpful assistant answering a multiple-choice science question. "
    "Reason step by step, then put your final answer (a single letter A, B, C, or D) "
    "within \\boxed{}."
)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def script_hashes():
    here = Path(__file__).resolve().parent
    return {n: file_hash(here / n) for n in ("mathplus_data.py", "mathplus_worker.py")}


def _pick(row, *names):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    raise KeyError(f"none of {names} in row {sorted(row)}")


def normalize_math(benchmark, row, idx):
    """AIME/AMC rows -> {id, problem, gold(number-as-string)}."""
    if benchmark == "aime24":
        problem = _pick(row, "Problem", "problem")
        answer = str(_pick(row, "Answer", "answer"))
        rid = str(row.get("ID", row.get("id", f"aime24_{idx}")))
    elif benchmark in ("aime25", "aime26"):
        problem = _pick(row, "problem", "Problem")
        answer = str(_pick(row, "answer", "Answer"))
        rid = str(row.get("id", f"{benchmark}_{idx}"))
    elif benchmark == "amc23":
        problem = _pick(row, "question", "Question", "problem")
        raw = _pick(row, "answer", "Answer")
        answer = str(int(float(raw)))  # dataset stores numeric answers as float
        rid = str(row.get("id", f"amc23_{idx}"))
    else:
        raise ValueError(benchmark)
    return {"id": rid, "problem": problem, "gold": answer}


def normalize_gpqa(row, idx):
    """GPQA diamond row -> {id, problem, choices[4], gold_letter}.

    Choice order is shuffled with a fixed seed (same rule as the server
    harness utils.py: rng = random.Random(42)) so every model sees the
    same layout.
    """
    question = _pick(row, "Question", "question")
    correct = _pick(row, "Correct Answer", "correct_answer").strip()
    wrongs = [_pick(row, f"Incorrect Answer {i}", f"incorrect_answer_{i}").strip()
              for i in (1, 2, 3)]
    choices = wrongs + [correct]
    rng = random.Random(42)
    rng.shuffle(choices)
    gold = "ABCD"[choices.index(correct)]
    rid = str(row.get("id", row.get("Id", f"gpqa-diamond_{idx}")))
    return {"id": rid, "problem": question, "choices": choices, "gold": gold}


def prompt(benchmark, item, protocol="simct"):
    """Build chat messages. Default protocol mirrors the SimCT AIME path
    (0-shot chat + QWEN boxed prompt). ``protocol="harness"`` mirrors the
    server lm-eval yaml (plain Question/Answer, no system prompt)."""
    if benchmark in MATH_BENCHES:
        if protocol == "harness":
            return [{"role": "user",
                     "content": f"Question: {item['problem']}\nAnswer:"}]
        return [{"role": "system", "content": QWEN_MATH_SYSTEM_PROMPT},
                {"role": "user", "content": item["problem"]}]
    if benchmark == "gpqa-diamond":
        body = item["problem"] + "\n" + "\n".join(
            f"({chr(65 + i)}) {c}" for i, c in enumerate(item["choices"]))
        if protocol == "harness":
            return [{"role": "user", "content": body}]
        return [{"role": "system", "content": GPQA_SYSTEM_PROMPT},
                {"role": "user", "content": body + "\nAnswer:"}]
    raise ValueError(benchmark)


def extract_boxed_letter(text):
    """Last \\boxed{X} single letter, else trailing standalone A-D."""
    if not text:
        return ""
    last = ""
    for m in re.finditer(r"\\boxed\{\s*([A-Da-d])\s*\}", text):
        last = m.group(1).upper()
    if last:
        return last
    m = re.search(r"\b([A-D])\b(?!.*\b[A-D]\b)", text)
    return m.group(1) if m else ""


def acquire(benchmark, cache, proxy):
    """Download the pinned file through the company proxy, verify, normalize."""
    repo, revision, name, _ = PINNED[benchmark]
    target = cache / (benchmark + Path(name).suffix)
    if not target.exists():
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        tmp = target.with_suffix(target.suffix + ".part")
        with opener.open(url, timeout=300) as response, tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        tmp.rename(target)
    sha = file_hash(target)
    if target.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(target).to_pylist()
    elif target.suffix == ".jsonl":
        rows = [json.loads(x) for x in target.read_text().splitlines() if x.strip()]
    elif target.suffix == ".csv":
        import csv
        rows = list(csv.DictReader(target.read_text().splitlines()))
    else:
        raise ValueError(target.suffix)
    items = []
    for i, row in enumerate(rows):
        if benchmark == "gpqa-diamond":
            item = normalize_gpqa(row, i)
        else:
            item = normalize_math(benchmark, row, i)
        item["messages"] = prompt(benchmark, item)
        item["messages_harness"] = prompt(benchmark, item, protocol="harness")
        items.append(item)
    if len(items) != COUNTS[benchmark] or len({x["id"] for x in items}) != len(items):
        raise ValueError(f"{benchmark}: count/IDs mismatch "
                         f"(got {len(items)}, want {COUNTS[benchmark]})")
    source = {"path": str(target), "sha256": sha, "repo": repo,
              "revision": revision, "remote": name}
    return items, source
