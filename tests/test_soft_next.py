import importlib.util
import json
from pathlib import Path
import pytest

root=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('softnext',root/'experiments/runai/queue_soft_next.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_failed_or_partial_soft_never_publishes_ready(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'BASE',tmp_path)
    run=tmp_path/'owner-tungks-0-1/soft50-followup/runs/run/checkpoint'
    run.mkdir(parents=True)
    (run/'run-summary.json').write_text(json.dumps(dict(status='stopped',optimizer_updates=35)))
    with pytest.raises(ValueError,match='did not complete'):
        m.prepare()


def test_no_summary_never_prepares_eval(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'BASE',tmp_path)
    with pytest.raises(ValueError,match='exactly one'):
        m.prepare()
