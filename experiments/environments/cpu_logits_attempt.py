"""Single managed retry after diagnosed full-batch CUDA-logit-buffer OOM.

Immutable original NeMo, explicitly recorded CPU transport overlay, unchanged
scientific config. Reuses exact data/projection/models; no downloads or installs.
Canary and config parity are mandatory before one bounded training invocation.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/'xtoken/scripts'))
from build_cpu_logits_overlay import build
from continue_target_data import sha
from resumable_target_data import json_atomic
from run_logged import run_logged
from run_target_after_setup import resource_gate
from xtoken_target import overrides, UPSTREAM_LOCK

OLD = 'target-4b-2048-b64-full-10steps-r4'
RUN = 'target-4b-2048-b64-full-10steps-cpu-r5'
OVERLAY = 'NeMo-RL-cpu-logits-r1'


def validate_previous(record, text):
    assert record['status'] == 'stopped' and record['exit_code'] == 1
    assert record['training_phase_invoked'] is True
    assert record['phases'][-1]['phase'] == 'train' and record['phases'][-1]['exit_code'] == 1
    assert [p['exit_code'] for p in record['phases'][1:-1]] == [0, 0, 0, 0]
    assert 'Tried to allocate 37.09 GiB' in text and 'ensure_teacher_ipc_buffer' in text


def memory_gate(packed=False):
    root = Path('/sys/fs/cgroup')
    limit = int((root/'memory.max').read_text())
    current = int((root/'memory.current').read_text())
    stats = dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
    headroom = limit - current + int(stats['inactive_file'])
    required = 60 if packed else 90
    assert headroom >= required * 1024**3, 'Insufficient reclaimable RAM for exact logit buffers plus overhead'
    return dict(limit_bytes=limit, current_bytes=current, reclaimable_headroom_bytes=headroom,
                projected_teacher_buffer_bytes=2*32*2048*151936*(2 if packed else 4))


def equivalent_config(old, new, run_name=RUN):
    # Only new artifact/output identity; symlinked data/projection bytes are exact.
    normalize = lambda value: json.dumps(value, sort_keys=True).replace(run_name, OLD)
    return normalize(old) == normalize(new)


def run(root, retry_platform_constants=False, retry_same_gpu_canary=False, retry_lossless_storage=False,
        retry_preoffload_cache=False):
    assert sum((retry_platform_constants, retry_same_gpu_canary, retry_lossless_storage,
                retry_preoffload_cache)) <= 1
    run_name = RUN.replace('cpu-r5', 'cpu-r7' if retry_same_gpu_canary else 'cpu-r6') if (retry_platform_constants or retry_same_gpu_canary) else RUN
    overlay_name = OVERLAY.replace('-r1', '-r3' if retry_same_gpu_canary else '-r2') if (retry_platform_constants or retry_same_gpu_canary) else OVERLAY
    if retry_lossless_storage:
        run_name, overlay_name = RUN.replace('cpu-r5', 'cpu-r8'), OVERLAY.replace('-r1','-r4')
    if retry_preoffload_cache:
        run_name, overlay_name = RUN.replace('cpu-r5', 'cpu-r9'), OVERLAY.replace('-r1','-r5')
    with (root/'.xtoken-managed.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = root/('sequence-'+run_name)
        out.mkdir(mode=0o700, exist_ok=False)
        state = dict(status='preflight', requested_optimizer_updates=10, training_phase_invoked=False,
            phases=[], started_at=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid(),
            predecessor=OLD, reason='Diagnosed 37.09GiB/rank CUDA full-batch logit buffer OOM',
            change='FP32 CPU memfd transport; upload only requested microbatch', automatic_replay=False)
        def save():
            state['updated_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
            json_atomic(out/'result.json', state)
        save()
        try:
            if retry_lossless_storage:
                failed = json.loads((root/('sequence-'+RUN.replace('cpu-r5','cpu-r7'))/'result.json').read_text())
                assert failed['status'] == 'stopped' and failed['exit_code'] == 1
                assert failed['training_phase_invoked'] and failed['phases'][-1]['phase'] == 'train'
                prior_ev = Path(failed['phases'][-1]['evidence'])
                assert prior_ev.resolve().is_relative_to(root/'evidence')
                failure = (prior_ev/'stdout.log').read_text()
                assert 'ray.exceptions.OutOfMemoryError' in failure and 'offload_after_refit' in failure
                state['operational_retry_of'] = str(prior_ev)
                state['change'] = 'Lossless FP32 zero-low-bit encoding in CPU memfd; reject any nonzero low bits'
            if retry_preoffload_cache:
                failed = json.loads((root/('sequence-'+RUN.replace('cpu-r5','cpu-r8'))/'result.json').read_text())
                assert failed['status'] == 'stopped' and failed['exit_code'] == 1
                assert failed['training_phase_invoked'] and failed['phases'][-1]['phase'] == 'train'
                prior_ev = Path(failed['phases'][-1]['evidence'])
                assert prior_ev.resolve().is_relative_to(root/'evidence')
                failure = (prior_ev/'stdout.log').read_text()
                assert 'torch.AcceleratorError: CUDA error: unknown error' in failure
                assert 'offload_after_refit' in failure
                state['operational_retry_of'] = str(prior_ev)
                state['change'] = 'Lossless FP32 storage plus CUDA cache eviction before teacher model offload'
            if retry_platform_constants or retry_same_gpu_canary:
                predecessor = RUN.replace('cpu-r5', 'cpu-r6') if retry_same_gpu_canary else RUN
                failed = json.loads((root/('sequence-'+predecessor)/'result.json').read_text())
                assert failed['status'] == 'stopped' and failed['exit_code'] == 1
                assert failed['training_phase_invoked'] is False
                assert len(failed['phases']) == 1 and failed['phases'][0]['phase'] == 'canary'
                prior_ev = Path(failed['phases'][0]['evidence'])
                assert prior_ev.resolve().is_relative_to(root/'evidence')
                expected_error = ('CUDA error: peer access is not supported between these two devices' if retry_same_gpu_canary
                                  else "AttributeError: module 'fcntl' has no attribute 'F_ADD_SEALS'")
                assert expected_error in (prior_ev/'stdout.log').read_text()
                state['operational_canary_retry_of'] = str(prior_ev)
                state['platform_fix'] = ('Same-GPU cross-process CUDA IPC baseline on both GPUs; no P2P requirement' if retry_same_gpu_canary
                                         else 'Linux UAPI seal constants fallback with readback verification')
            previous = root/('sequence-'+OLD)
            record = json.loads((previous/'result.json').read_text())
            evidence = Path(record['phases'][-1]['evidence'][0])
            assert evidence.resolve().is_relative_to(root/'evidence')
            assert json.loads((evidence/'result.json').read_text())['exit_code'] == 1
            validate_previous(record, (evidence/'stdout.log').read_text())
            fingerprints = json.loads((previous/'source-fingerprints.json').read_text())
            for name, value in fingerprints.items():
                path = Path(name)
                assert path.is_relative_to(root/'NeMo-RL') or path.is_relative_to(HERE.parent)
                assert sha(path) == value, 'Predecessor source drift: '+path.name
            old = root/OLD
            models = json.loads((old/'models.json').read_text())
            assert models['complete'] and all(m['weights_verified'] for m in models['models'])
            for model in models['models']:
                for item in model['files']:
                    path = Path(model['snapshot'])/item['name']
                    assert path.resolve().is_relative_to(root/'hf')
                    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
            data = json.loads((old/'data/manifest.json').read_text())
            assert data['revision'] == '13fa979be2e7f7e62913eee0ec5e97c8fd6e24af'
            assert sha(old/'data/formal-logic-prefix.parquet') == data['parquet_sha256']
            projection = json.loads((old/'projection-full/manifest.json').read_text())
            assert projection['valid']
            for item in projection['artifacts']:
                assert sha(old/'projection-full'/item['file']) == item['sha256']
            assert sha(root/'NeMo-RL/uv.lock') == UPSTREAM_LOCK
            assert sha(HERE/'xtoken/uv.lock') == '145d512cf6e56deec88eacfde4159ba97fd55496a26e26d5aec8d33b7ba357cb'
            resource_gate()
            packed = retry_lossless_storage or retry_preoffload_cache
            state['memory_gate'] = memory_gate(packed=packed)
            assert shutil.disk_usage(root).free > 3*1024**3
            source = root/overlay_name
            helper = HERE/('cpu_logits_packed_transport.py' if packed else 'cpu_logits_transport.py')
            manifest = build(root/'NeMo-RL', source, helper)
            target = root/run_name
            target.mkdir(mode=0o700, exist_ok=False)
            shutil.copyfile(old/'models.json', target/'models.json')
            (target/'data').symlink_to(old/'data', target_is_directory=True)
            (target/'projection-full').symlink_to(old/'projection-full', target_is_directory=True)
            json_atomic(target/'reuse-and-overlay.json', dict(predecessor=str(old), overlay=manifest,
                training_restarts_from_base=True, original_artifacts_read_only=True,
                model_transfer_repeated=False, projection_recomputed=False, data_recomputed=False))
            fingerprints.update({str(p): sha(p) for p in HERE.glob('*.py')})
            fingerprints.update({str(p): sha(p) for p in source.rglob('*')
                                 if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')})
            json_atomic(out/'source-fingerprints.json', fingerprints)
            env = dict(PATH='/usr/local/bin:/usr/bin:/bin', HOME='/root', LANG='C.UTF-8',
                PYTHONUNBUFFERED='1', UV_CACHE_DIR=str(root/'uv-cache'), UV_NO_CACHE='false',
                UV_LINK_MODE='hardlink', UV_PYTHON_INSTALL_DIR=str(root/'python'),
                UV_PROJECT_ENVIRONMENT=str(root/'xtoken/venv'), HF_HOME=str(root/'hf'),
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                WANDB_MODE='disabled', PYTHONPATH=str(source), OMP_NUM_THREADS='4',
                CUDA_VISIBLE_DEVICES='0,1', NEMO_RL_VENV_DIR=str(root/'worker-venvs'),
                NEMO_RL_PY_EXECUTABLES_SYSTEM='1', RAY_USAGE_STATS_ENABLED='0',
                NCCL_CUMEM_HOST_ENABLE='0', TORCH_CUDA_ARCH_LIST='8.6')
            base = ['uv', 'run', '--project', str(HERE/'xtoken'), '--locked', '--no-sync', 'python']
            student, teacher = [m['snapshot'] for m in models['models']]
            settings = overrides(target, student, teacher, 10)
            commands = [
                ('canary', base+['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                                 str(HERE/'cpu_logits_canary.py')], 300),
                ('config', base+[str(HERE/'validate_xtoken_target.py'), '--expected-steps', '10', *settings], 300),
                ('train', base+[str(source/'examples/run_xtoken_off_policy_distillation.py'), *settings], 1800)]
            for phase, command, timeout in commands:
                assert all(sha(Path(p)) == h for p,h in fingerprints.items()), 'Frozen source drift'
                if phase != 'config':
                    resource_gate()
                if phase == 'train':
                    state['memory_gate_before_train'] = memory_gate(packed=packed)
                    assert equivalent_config(json.loads((old/'config-resolved.json').read_text()),
                                             json.loads((target/'config-resolved.json').read_text()), run_name)
                    json_atomic(target/'config-equivalence.json', dict(exact_except_run_paths=True,
                        previous_sha256=sha(old/'config-resolved.json'), new_sha256=sha(target/'config-resolved.json'),
                        native_lock_sha256=sha(HERE/'xtoken/uv.lock'), overlay=overlay_name))
                    state['training_phase_invoked'] = True
                state.update(status='running_phase', current_phase=phase)
                save()
                rc, ev = run_logged(command, cwd=source, root=root/'evidence',
                                    name='xtoken-'+run_name.split('10steps-')[1]+'-'+phase, timeout=timeout, env=env)
                state['phases'].append(dict(phase=phase, exit_code=rc, evidence=str(ev)))
                save()
                assert rc == 0, phase+' failed; preserve evidence and diagnose before new attempt'
                if phase == 'canary':
                    text = (ev/'stdout.log').read_text()
                    assert text.count('"cross_process_values_bitwise": true') == 4
            state.update(status='training_process_complete_pending_metric_review', exit_code=0)
        except Exception as exc:
            state.update(status='stopped', exit_code=1, error_type=type(exc).__name__, error=str(exc))
        finally:
            state['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
            save()
        return state['exit_code']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--retry-platform-constants', action='store_true')
    parser.add_argument('--retry-same-gpu-canary', action='store_true')
    parser.add_argument('--retry-lossless-storage', action='store_true')
    parser.add_argument('--retry-preoffload-cache', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    assert root == Path('/workspace/vdt-runtime')
    os.umask(0o077)
    os.environ.clear()
    os.environ.update(PATH='/usr/local/bin:/usr/bin:/bin', HOME='/root', LANG='C.UTF-8', PYTHONUNBUFFERED='1')
    sys.exit(run(root, retry_platform_constants=args.retry_platform_constants,
                 retry_same_gpu_canary=args.retry_same_gpu_canary,
                 retry_lossless_storage=args.retry_lossless_storage,
                 retry_preoffload_cache=args.retry_preoffload_cache))
