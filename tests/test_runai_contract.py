import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_preflight_configuration_and_input_hashes(tmp_path):
    for name in ('student','teacher'):
        (tmp_path/name).mkdir();(tmp_path/name/'config.json').write_text('{}')
    data=tmp_path/'prompts.parquet'; data.write_bytes(b'fixture')
    env=dict(os.environ,MP_PREFLIGHT_ONLY='1',MP_STUDENT_PATH=str(tmp_path/'student'),
             MP_TEACHER_PATH=str(tmp_path/'teacher'),MP_DATASET_PATH=str(data),CUDA_VISIBLE_DEVICES='0',
             MP_SEED='17',MP_MAX_SPAN_LENGTH='4',MP_FIXED_SPAN_LENGTH='4',MP_ALGORITHM='mp_opd')
    result=subprocess.run([sys.executable,str(ROOT/'experiments/runai/run_single_gpu.py'),'fixed','50',str(tmp_path/'out')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    config=json.loads((tmp_path/'out/launch-config.json').read_text())
    assert config['options']['seed']==17
    assert config['options']['micro_train_batch_size']==4
    assert config['options']['mp_opd_fixed_span_length']==4
    assert config['contract']['scheduler_horizon']==312
    assert config['contract']['execution_updates']==50
    assert config['options']['save_steps']==20
    assert len(config['dataset_sha256'])==64
    assert config['models_sha256']['student']['config.json']
    # Check manifest changes on data drift; no Ray or model execution.
    data.write_bytes(b'changed')
    result=subprocess.run([sys.executable,str(ROOT/'experiments/runai/run_single_gpu.py'),'random','5',str(tmp_path/'out2')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    config2=json.loads((tmp_path/'out2/launch-config.json').read_text())
    assert config2['dataset_sha256']!=config['dataset_sha256']
    env['MP_ALGORITHM']='span_ctkd'
    result=subprocess.run([sys.executable,str(ROOT/'experiments/runai/run_single_gpu.py'),'atomic','5',str(tmp_path/'out3')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    config3=json.loads((tmp_path/'out3/launch-config.json').read_text())
    assert config3['contract']['parity'] is None


def test_soft_preflight_propagates_adam_moment_offload(tmp_path):
    for name in ('student', 'teacher'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'config.json').write_text('{}')
    data = tmp_path / 'prompts.parquet'
    data.write_bytes(b'fixture')
    energy = tmp_path / 'energy.pt'
    meta = tmp_path / 'meta.parquet'
    energy.write_bytes(b'energy')
    meta.write_bytes(b'meta')
    env = dict(
        os.environ,
        MP_PREFLIGHT_ONLY='1',
        MP_STUDENT_PATH=str(tmp_path / 'student'),
        MP_TEACHER_PATH=str(tmp_path / 'teacher'),
        MP_DATASET_PATH=str(data),
        MP_ENERGY_CHECKPOINT=str(energy),
        MP_META_PATH=str(meta),
        MP_ALTERNATING='1',
        MP_ATTN_IMPLEMENTATION='eager',
        MP_OFFLOAD_ADAM_MOMENTS='1',
        CUDA_VISIBLE_DEVICES='0',
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / 'experiments/runai/run_single_gpu.py'), 'soft', '2', str(tmp_path / 'out')],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / 'out/launch-config.json').read_text())
    assert config['options']['mp_opd_offload_adam_moments'] is True
    assert 'EFFECTIVE_MP_OPD_OFFLOAD_ADAM_MOMENTS=true' in result.stdout


def test_full_soft_micro1_meta4_admission_preserves_production_contract(tmp_path):
    for name in ('student', 'teacher'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'config.json').write_text('{}')
    data = tmp_path / 'prompts.parquet'
    data.write_bytes(b'fixture')
    energy = tmp_path / 'energy.pt'
    meta = tmp_path / 'meta.parquet'
    energy.write_bytes(b'energy')
    meta.write_bytes(b'meta')
    env = dict(
        os.environ,
        MP_PREFLIGHT_ONLY='1',
        MP_STUDENT_PATH=str(tmp_path / 'student'),
        MP_TEACHER_PATH=str(tmp_path / 'teacher'),
        MP_DATASET_PATH=str(data),
        MP_ENERGY_CHECKPOINT=str(energy),
        MP_META_PATH=str(meta),
        MP_ALTERNATING='1',
        MP_ATTN_IMPLEMENTATION='eager',
        MP_OFFLOAD_ADAM_MOMENTS='1',
        MP_MICRO_TRAIN_BATCH_SIZE='1',
        MP_META_MICRO_BATCH_SIZE='4',
        MP_ALGORITHM='mp_opd',
        MP_MAX_SPAN_LENGTH='2',
        MP_FIXED_SPAN_LENGTH='2',
        MP_ENERGY_EVERY='1',
        MP_ENERGY_LR='0.001',
        CUDA_VISIBLE_DEVICES='0',
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / 'experiments/runai/run_single_gpu.py'), 'soft', '312', str(tmp_path / 'out')],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'ADMISSION_PASS: exact soft alternating micro1/meta4 full-run contract' in result.stdout
    config = json.loads((tmp_path / 'out/launch-config.json').read_text())
    options = config['options']
    assert config['contract']['execution_updates'] == 312
    assert config['contract']['scheduler_horizon'] == 312
    assert options['train_batch_size'] == 64
    assert options['micro_train_batch_size'] == 1
    assert options['mp_opd_meta_batch_size'] == 16
    assert options['mp_opd_meta_microbatch_size'] == 4
    assert options['max_len'] == 4096
    assert options['generate_max_len'] == 4096
    assert options['attn_implementation'] == 'eager'
    assert options['mp_opd_offload_adam_moments'] is True

OPERATIONAL_KEYS = (
    'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'CUDA_LAUNCH_BLOCKING',
    'NCCL_CUMEM_HOST_ENABLE', 'NCCL_IB_DISABLE', 'NCCL_NET_GDR_LEVEL', 'NCCL_P2P_DISABLE',
    'OMP_NUM_THREADS', 'RAY_USAGE_STATS_ENABLED', 'TOKENIZERS_PARALLELISM',
)


def _soft_fixture(tmp_path):
    for name in ('student', 'teacher'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'config.json').write_text('{}')
    data = tmp_path / 'prompts.parquet'
    data.write_bytes(b'fixture')
    energy = tmp_path / 'energy.pt'
    meta = tmp_path / 'meta.parquet'
    energy.write_bytes(b'energy')
    meta.write_bytes(b'meta')
    env = {k: v for k, v in os.environ.items() if k not in OPERATIONAL_KEYS}
    env.update(
        MP_PREFLIGHT_ONLY='1',
        MP_STUDENT_PATH=str(tmp_path / 'student'),
        MP_TEACHER_PATH=str(tmp_path / 'teacher'),
        MP_DATASET_PATH=str(data),
        MP_ENERGY_CHECKPOINT=str(energy),
        MP_META_PATH=str(meta),
        MP_ALTERNATING='1',
        MP_ATTN_IMPLEMENTATION='eager',
        MP_OFFLOAD_ADAM_MOMENTS='1',
        CUDA_VISIBLE_DEVICES='0',
    )
    return env


def test_operational_environment_is_empty_by_default(tmp_path):
    env = _soft_fixture(tmp_path)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'experiments/runai/run_single_gpu.py'), 'soft', '2', str(tmp_path / 'out')],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / 'out/launch-config.json').read_text())
    assert config['operational_environment'] == {}
    assert 'EFFECTIVE_OPERATIONAL_ENV={}' in result.stdout
    assert 'OPERATIONAL_OVERRIDE' not in result.stdout


def test_allocator_override_is_recorded_and_announced(tmp_path):
    env = _soft_fixture(tmp_path)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'experiments/runai/run_single_gpu.py'), 'soft', '2', str(tmp_path / 'out')],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / 'out/launch-config.json').read_text())
    assert config['operational_environment']['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True'
    assert 'OPERATIONAL_OVERRIDE expandable_segments=True' in result.stdout
    assert 'expandable_segments:True' in result.stdout


def test_resume_tolerates_a_manifest_written_without_the_operational_record(tmp_path):
    env = _soft_fixture(tmp_path)
    out = tmp_path / 'out'
    first = subprocess.run(
        [sys.executable, str(ROOT / 'experiments/runai/run_single_gpu.py'), 'soft', '2', str(out)],
        env=env, capture_output=True, text=True,
    )
    assert first.returncode == 0, first.stderr
    manifest = json.loads((out / 'launch-config.json').read_text())
    manifest.pop('operational_environment')  # a manifest written before this field existed
    (out / 'launch-config.json').write_text(json.dumps(manifest))
    (out / 'checkpoints').mkdir()
    (out / 'checkpoints/latest.json').write_text('{}')
    resumed = subprocess.run(
        [sys.executable, str(ROOT / 'experiments/runai/run_single_gpu.py'), 'soft', '2', str(out)],
        env=dict(env, MP_RESUME='1'), capture_output=True, text=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    attempts = sorted(out.glob('resume-attempt-*.json'))
    assert attempts, 'resume must record what it ran under'
    recorded = json.loads(attempts[-1].read_text())['operational_environment']
    assert recorded == {}

def test_diagnostic_max_len_is_applied_and_bounded(tmp_path):
    env = _soft_fixture(tmp_path)
    env['MP_MAX_LEN'] = '2048'
    short = subprocess.run(
        [sys.executable, str(ROOT / 'experiments/runai/run_single_gpu.py'), 'soft', '2', str(tmp_path / 'out')],
        env=env, capture_output=True, text=True,
    )
    assert short.returncode == 0, short.stderr
    config = json.loads((tmp_path / 'out/launch-config.json').read_text())
    assert config['options']['max_len'] == 2048
    full = subprocess.run(
        [sys.executable, str(ROOT / 'experiments/runai/run_single_gpu.py'), 'soft', '312', str(tmp_path / 'out2')],
        env=env, capture_output=True, text=True,
    )
    assert full.returncode != 0
    assert 'diagnostic-only knob' in full.stderr

def test_launcher_binds_and_announces_the_effective_max_len():
    """The manifest must not be able to disagree with what the rollout actually used."""
    import ast
    source = ROOT / 'experiments/runai/run_single_gpu.py'
    tree = ast.parse(source.read_text())
    init = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, 'attr', '') == 'init_args']
    assert len(init) == 1
    # init_args raises max_len back to prompt_max_len + generate_max_len, so the launcher
    # must rebind the effective value afterwards and print it for the run log.
    rebind = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Attribute) and t.attr == 'max_len' for t in node.targets)]
    printed = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and getattr(node.func, 'id', '') == 'print'
               and 'EFFECTIVE_DATA_MAX_LEN' in ast.dump(node)]
    assert rebind and printed
    assert min(node.lineno for node in rebind) > init[0].lineno
    assert min(node.lineno for node in printed) > init[0].lineno
