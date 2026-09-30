"""Pinned public teacher, deterministic data prefix, and full-projection audit."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlencode
from urllib.request import urlopen

TEACHER = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
WEIGHTS = {
    'model-00001-of-00003.safetensors': '328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223',
    'model-00002-of-00003.safetensors': '6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5',
    'model-00003-of-00003.safetensors': 'e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def models(root, target):
    assert shutil.disk_usage(root).free > 12 * 1024**3, '12 GiB model/storage headroom required'
    old = json.loads((root/'previous-models.json').read_text())
    student = old['models'][0]
    assert student['repo'] == 'meta-llama/Llama-3.2-1B'
    assert student['revision'] == '4e20de362430cd3b72f300e6b0f18e50e7166e08'
    for item in student['files']:
        path = Path(student['snapshot'])/item['name']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    # Public teacher only; no token, no default credential lookup.
    env = {k: v for k, v in os.environ.items() if not any(
        word in k.upper() for word in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY'))}
    env.update(HF_HOME=str(root/'hf'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    env.pop('HF_HUB_OFFLINE', None)
    env.pop('TRANSFORMERS_OFFLINE', None)
    filenames = ['config.json', 'generation_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'model.safetensors.index.json', *WEIGHTS]
    command = [str(root/'xtoken/venv/bin/hf'), 'download', TEACHER,
               *filenames, '--revision', REVISION, '--cache-dir', str(root/'hf/hub'),
               '--max-workers', '3', '--quiet']
    subprocess.run(command, env=env, check=True, timeout=1500)
    snapshot = root/'hf/hub/models--Qwen--Qwen3-4B/snapshots'/REVISION
    records = []
    for name in filenames:
        path = snapshot/name
        checksum = sha(path)
        if name in WEIGHTS:
            assert checksum == WEIGHTS[name], f'Weight checksum mismatch: {name}'
        records.append(dict(name=name, bytes=path.stat().st_size, sha256=checksum))
    index = json.loads((snapshot/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'].values()) == set(WEIGHTS)
    config = json.loads((snapshot/'config.json').read_text())
    assert config['model_type'] == 'qwen3' and config['vocab_size'] == 151936
    teacher = dict(role='teacher', repo=TEACHER, revision=REVISION,
                   snapshot=str(snapshot), weights_verified=True, files=records)
    save(target/'models.json', dict(complete=True, models=[student, teacher],
         credentials_used_by_this_validation=False, teacher_public=True,
         student_matches_previous_verified_manifest=True))
    print(json.dumps(dict(models_verified=True, teacher=TEACHER, revision=REVISION)), flush=True)


def required_packs(steps):
    assert steps in (3, 10)
    return (steps + 1) * 64


def data(target, steps):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = target/'data'
    out.mkdir(exist_ok=False)
    dataset = 'nvidia/Nemotron-Pretraining-Specialized-v1.1'
    subset = 'Nemotron-Pretraining-Formal-Logic'
    expected = '13fa979be2e7f7e62913eee0ec5e97c8fd6e24af'
    rows, packed, buffered = [], 0, 0
    # Contiguous deterministic prefix, no synthetic repetition/content selection.
    # One extra complete batch beyond the authorized update count, without replay.
    minimum_packs = required_packs(steps)
    query = urlencode(dict(dataset=dataset))
    with urlopen('https://datasets-server.huggingface.co/parquet?' + query, timeout=60) as response:
        assert response.headers.get('X-Revision') == expected, 'Viewer revision changed'
        artifacts = json.load(response)['parquet_files']
    candidates = [item for item in artifacts
                  if item['config'] == subset and item['split'] == 'train']
    assert len(candidates) == 1, 'Expected exactly one pinned train parquet shard'
    artifact = candidates[0]
    source = out/'.source.parquet.pending'
    with urlopen(artifact['url'], timeout=900) as response, source.open('xb') as stream:
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
    assert source.stat().st_size == artifact['size'], 'Parquet transport size mismatch'
    source_sha256 = sha(source)
    parquet_source = pq.ParquetFile(source)
    for batch in parquet_source.iter_batches(batch_size=256, columns=['text']):
        for text in batch.column('text').to_pylist():
            assert isinstance(text, str) and text
            rows.append(dict(text=text))
            buffered += len(text)
            if buffered >= 16384:
                packed += 1
                buffered = 0
            if packed >= minimum_packs:
                break
        print(json.dumps(dict(raw_rows=len(rows), complete_packs=packed)), flush=True)
        if packed >= minimum_packs:
            break
    assert packed >= minimum_packs, 'Insufficient prefix; do not repeat rows or reduce batch'
    parquet = out/'formal-logic-prefix.parquet'
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    source.unlink()
    save(out/'manifest.json', dict(dataset=dataset, subset=subset, revision=expected,
         raw_rows=len(rows), complete_packs=packed, characters_per_sample=16384,
         optimizer_updates=steps, minimum_presentations=steps * 64,
         requested_complete_packs=minimum_packs,
         parquet_sha256=sha(parquet), source_filename=artifact['filename'],
         source_bytes=artifact['size'], source_sha256=source_sha256,
         transport='Dataset Viewer pinned parquet; contiguous row-order prefix',
         fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),
         scope='bounded workload test prefix, not full training corpus'))


def projection_audit(target):
    import torch
    manifest = json.loads((target/'models.json').read_text())
    student, teacher = [m['snapshot'] for m in manifest['models']]
    reports = []
    for name, columns in [('prep32_special.pt', 32),
                          ('prep32_special_exact_map_remapped.pt', 32), ('runtime_top4.pt', 4)]:
        path = target/'projection-full'/name
        obj = torch.load(path, map_location='cpu', weights_only=True)
        indices, weights = obj['indices'], obj['likelihoods']
        assert tuple(indices.shape) == tuple(weights.shape) == (128256, columns)
        assert not indices.is_floating_point()
        assert torch.isfinite(weights).all() and (weights >= 0).all()
        assert (indices >= -1).all() and (indices < 151936).all()
        assert obj['student_model_id'] == student and obj['teacher_model_id'] == teacher
        assert obj['enable_scale_trick'] is True
        assert (weights[indices == -1] == 0).all()
        sums = weights.sum(1)
        reports.append(dict(file=name, sha256=sha(path), shape=list(indices.shape),
             indices_dtype=str(indices.dtype), weights_dtype=str(weights.dtype),
             index_min=int(indices.min()), index_max=int(indices.max()),
             weights_min=float(weights.min()), weights_max=float(weights.max()),
             zero_rows=int((sums == 0).sum()), row_sum_min=float(sums.min()),
             row_sum_max=float(sums.max()), finite=True))
    save(target/'projection-full/manifest.json', dict(valid=True, artifacts=reports,
         upstream_commit='13a10647ebbf0f940d2b06ea41800b3f2fb46099',
         pipeline=['multitoken top32 reverse+scale+special', 'reapply exact map',
                   'sort/cut runtime top4 preserving scale metadata'],
         embedding_model_used=False, equivalence_to_smoke_not_claimed=True))
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['models', 'data', 'projection-audit'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--steps', type=int, choices=(3, 10))
    args = parser.parse_args()
    if args.phase == 'models':
        models(args.root, args.target)
    elif args.phase == 'data':
        if args.steps is None:
            parser.error('--steps is required for data preparation')
        data(args.target, args.steps)
    else:
        projection_audit(args.target)
