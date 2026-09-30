"""Real advisory-lock concurrency, synthetic scoring, no GPU or network."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/evaluation'))
import eval_queue as Q


def test_two_scoring_workers_share_cells_without_duplicate_writes(tmp_path):
    data=tmp_path/'data';data.write_text('{}')
    plan=dict(profile=Q.D.PROFILE,source={},seeds=[42,43],
        data={'gsm8k':dict(path=str(data),sha256=Q.E.file_hash(data),count=1)},
        jobs=[dict(id='a',checkpoint={'sha256':'cp'})])
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan))
    (tmp_path/'state.json').write_text(json.dumps(dict(qualification=[],deadline=time.time()+20)))
    (tmp_path/'generation-state.json').write_text(json.dumps(dict(jobs={'a':{'status':'completed'}})))
    for seed in plan['seeds']:
        cell=tmp_path/'cells/a/gsm8k'/str(seed);cell.mkdir(parents=True)
        response=cell/'responses.jsonl';response.write_text('{}\n')
        (cell/'generation-complete.json').write_text(json.dumps(dict(
            responses_sha256=Q.E.file_hash(response),contract={'server':{}},count=1)))
    args=argparse.Namespace(plan=p,score_python=sys.executable)
    barrier=threading.Barrier(2)
    scored=[];errors=[]
    def score(root,plan,ph,job,benchmark,seed,*rest):
        scored.append(seed)
        barrier.wait(timeout=3)  # Both workers must claim different cells concurrently.
        (root/'cells/a/gsm8k'/str(seed)/'metrics.json').write_text('{}')
    def worker():
        try: Q.score_spool(args)
        except Exception as exc: errors.append(exc)
    with patch.object(Q.D,'script_hashes',return_value={}), patch.object(Q,'preflight',return_value=[]), \
         patch.object(Q.E,'CAPS',{'gsm8k':1}), patch.object(Q,'cell_contract',return_value={'server':{}}), \
         patch.object(Q,'run_cell',side_effect=score):
        threads=[threading.Thread(target=worker) for _ in range(2)]
        for t in threads:t.start()
        for t in threads:t.join(timeout=12)
        assert all(not t.is_alive() for t in threads)
    assert not errors
    assert sorted(scored)==[42,43]


def test_shared_worker_lock_refuses_legacy_exclusive_coordinator(tmp_path):
    path=tmp_path/'scoring.lock'
    with Q.locked(path) as legacy:
        assert legacy is not None
        with Q.locked(path,blocking=False,shared=True) as new:
            assert new is None
    with Q.locked(path,shared=True):
        with Q.locked(path,blocking=False,shared=True) as second:
            assert second is not None


def test_cell_claim_released_after_failure(tmp_path):
    path=tmp_path/'score.lock'
    try:
        with Q.locked(path):
            raise RuntimeError('worker interrupted')
    except RuntimeError:
        pass
    with Q.locked(path,blocking=False) as claim:
        assert claim is not None


def test_retry_archives_cell_error_and_keeps_scores(tmp_path):
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(jobs=[{'id':'a'}],seeds=[42])))
    cell=tmp_path/'cells/a/gsm8k/42';cell.mkdir(parents=True)
    (cell/'score-error.json').write_text('{"error":"fixture"}')
    scores=cell/'scores.jsonl';scores.write_text('existing evidence\n')
    with patch.object(Q.E,'CAPS',{'gsm8k':1}):
        Q.retry_scoring(argparse.Namespace(plan=plan))
    assert not (cell/'score-error.json').exists()
    assert len(list(cell.glob('score-error-ack-*.json')))==1
    assert scores.read_text()=='existing evidence\n'
