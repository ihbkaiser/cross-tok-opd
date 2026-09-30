import importlib.util
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location('followup', ROOT/'experiments/runai/submit_followup.py')
module = importlib.util.module_from_spec(loader)
loader.loader.exec_module(module)


def test_new_simct_does_not_inherit_borrowed_deadline():
    old = dict(env={'MP_TRAIN_STOP_AT':'100','MP_ENERGY_CHECKPOINT':'old'},deadline=100,
               timeout_seconds=5,dependencies=['old'])
    new = module.specification(old,'simct','GPU-test',Path('/tmp/output'))
    assert 'deadline' not in new and 'MP_TRAIN_STOP_AT' not in new['env']
    assert new['env']['MP_SEED']=='43'
    assert new['env']['MP_ALGORITHM']=='span_ctkd'
    assert new['env']['MP_ATTN_IMPLEMENTATION']=='sdpa'
    assert new['argv'][-2:]==['atomic','0']
    assert old['deadline']==100 and old['env']['MP_TRAIN_STOP_AT']=='100'


def test_soft_keeps_diagnostic_capture_and_finite_update_limit():
    new = module.specification(dict(env={}), 'soft','GPU-test',Path('/tmp/output'))
    assert new['argv'][-2:]==['soft','50']
    assert new['env']['MP_SEED']=='42'
    assert new['env']['MP_ATTN_IMPLEMENTATION']=='eager'
    assert new['env']['MP_PARITY_CAPTURE_DIR'].endswith('/captures')


def test_lcb_audit_checks_hashes_and_reports_overlapping_failures(tmp_path):
    import pytest
    loader = importlib.util.spec_from_file_location('audit',ROOT/'experiments/runai/audit_lcb_outputs.py')
    audit = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(audit)
    response = dict(id='a',response={'choices':[dict(finish_reason='length',message={'content':'def ('})]})
    score = dict(id='a',passed=False,timeout=True)
    metrics = dict(count=1)
    for name,row in [('responses',response),('scores',score)]:
        raw = (json.dumps(row)+'\n').encode()
        (tmp_path/(name+'.jsonl')).write_bytes(raw)
        metrics[name+'_sha256'] = hashlib.sha256(raw).hexdigest()
    (tmp_path/'metrics.json').write_text(json.dumps(metrics))
    helpers = dict(strip_thinking_content=lambda t:(t,False,False),_extract_code_block=lambda t:t)
    counts = audit.audit(tmp_path,helpers)['counts']
    assert counts['syntax_error_failed']==counts['finish_length_failed']==counts['outer_timeout_failed']==1
    (tmp_path/'scores.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        audit.audit(tmp_path,helpers)
