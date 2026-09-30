import importlib.util
import json
from pathlib import Path
import tarfile

spec = importlib.util.spec_from_file_location('collector', Path(__file__).parents[1]/'experiments/runai/collect_campaign.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_export_preserves_partial_and_excludes_weights_and_secrets(tmp_path):
    root = tmp_path/'campaign'
    run = root/'runs'/'one'
    ckpt = run/'checkpoint'/'step40'
    ckpt.mkdir(parents=True)
    (ckpt/'model.safetensors').write_bytes(b'weights')
    (ckpt/'tokenizer.json').write_text('{"model": {}}')
    (ckpt/'vocab.json').write_text('{}')
    (run/'train.log').write_text('step [41/312]')
    (run/'secret.json').write_text('private')
    (run/'responses.jsonl').write_text('{}\n')
    (root/'source').mkdir()
    (root/'source'/'config.json').write_text('{}')
    (root/'linked').symlink_to(run, target_is_directory=True)
    report = m.inventory([root])
    assert len(report['files']) == 1
    assert report['checkpoints'][0]['validated'] is False
    archive = tmp_path/'evidence.tar.gz'
    m.export(report, archive)
    with tarfile.open(archive) as tar:
        assert tar.getnames() == ['root0/runs/one/train.log', 'manifest.json']
        manifest = json.load(tar.extractfile('manifest.json'))
        assert len(manifest['files'][0]['sha256']) == 64
    assert len(m.inventory([root], responses=True)['files']) == 2

def test_missing_root_and_live_change_are_explicit(tmp_path):
    (tmp_path/'train.log').write_text('first')
    report = m.inventory([tmp_path, tmp_path/'missing'])
    (tmp_path/'train.log').write_text('changed log')
    m.export(report, tmp_path/'export.tar.gz')
    assert report['files'][0]['changed_during_collection']
    assert not report['roots'][1]['exists']
