"""Build an explicitly recorded operational overlay; original NeMo is immutable."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

PINS = {
    'nemo_rl/models/policy/utils.py': 'becfc5e725024fa0424970a47d3a55a3872b67fb0717d6c85f7ab8ba3498b5fb',
    'nemo_rl/algorithms/x_token/loss_utils.py': 'a0ac80eb3bd47f68377545699a2ec611edffd14680b81121fd9cc096d920f987',
    'nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py': '942d95e767dc9f5863cad84bb02df461dd24860353cf2c553be01c45ec8f29af',
}
ALLOC_OLD = '    needs_realloc = (\n'
ALLOC_NEW = '''    # VDT operational CPU transport: preserve all full-precision logits.
    from nemo_rl.models.policy.vdt_cpu_logits import ensure_cpu_buffer
    return ensure_cpu_buffer(storage, handle,
                             (num_microbatches, batch_size, seq_len, vocab_size), dtype)

    needs_realloc = (
'''
REBUILD_OLD = '    func = rebuild_cuda_tensor\n'
REBUILD_NEW = '''    from nemo_rl.models.policy.vdt_cpu_logits import TAG, rebuild_cpu_buffer
    if cuda_ipc_handle and cuda_ipc_handle[0] == TAG:
        return rebuild_cpu_buffer(cuda_ipc_handle)
    func = rebuild_cuda_tensor
'''
VIEW_OLD = '    return src_full[buf_idx, : len(chosen), seq_lo:seq_hi, :full_vocab_size]\n'
VIEW_NEW = '''    from nemo_rl.models.policy.vdt_cpu_logits import transport_view
    view = src_full[buf_idx, : len(chosen), seq_lo:seq_hi, :full_vocab_size]
    return transport_view(view, device)
'''
CACHE_OLD = '''    def offload_after_refit(self) -> None:
        """Offload as much as possible on the CPU."""
        self.model = self.move_to_cpu(self.model)
'''
CACHE_NEW = '''    def offload_after_refit(self) -> None:
        """Offload as much as possible on the CPU."""
        # VDT operational fix: full-vocab temporaries have been released but
        # remain cached. Free them before model D2H/offload and allocator wakeup.
        gc.collect()
        torch.cuda.empty_cache()
        self.model = self.move_to_cpu(self.model)
'''


def transform(relative, source):
    if relative.endswith('/utils.py'):
        replacements = [(ALLOC_OLD, ALLOC_NEW), (REBUILD_OLD, REBUILD_NEW)]
    elif relative.endswith('/loss_utils.py'):
        replacements = [(VIEW_OLD, VIEW_NEW)]
    else:
        replacements = [(CACHE_OLD, CACHE_NEW)]
    for old, new in replacements:
        assert source.count(old) == 1, 'Expected exact unique upstream anchor'
        source = source.replace(old, new)
    ast.parse(source)
    return source


def build(source, destination, helper):
    assert source.is_absolute() and destination.is_absolute()
    assert not destination.exists() and not destination.is_relative_to(source)
    originals = {}
    for relative, expected in PINS.items():
        raw = (source/relative).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, 'Upstream drift'
        originals[relative] = raw
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    changes = []
    for relative, raw in originals.items():
        output = transform(relative, raw.decode()).encode()
        (destination/relative).write_bytes(output)
        changes.append(dict(path=relative, before=hashlib.sha256(raw).hexdigest(),
                            after=hashlib.sha256(output).hexdigest()))
        assert (source/relative).read_bytes() == raw
    helper_relative = 'nemo_rl/models/policy/vdt_cpu_logits.py'
    shutil.copyfile(helper, destination/helper_relative)
    allowed = set(PINS) | {helper_relative}
    for path in destination.rglob('*'):
        if path.is_file() and str(path.relative_to(destination)) not in allowed:
            original = source/path.relative_to(destination)
            assert original.is_file() and original.read_bytes() == path.read_bytes()
    manifest = dict(upstream_commit='13a10647ebbf0f940d2b06ea41800b3f2fb46099',
        upstream_source_unchanged=True, operational_overlay=True, changes=changes,
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        precision_changed=False, loss_changed=False, optimizer_changed=False,
        allocator_cache_evicted_before_teacher_offload=True,
        storage_encoding=('zero-low16 FP32 bit packing; reject nonzero low bits, exact FP32 decode'
                          if helper.name == 'cpu_logits_packed_transport.py' else 'unpacked FP32'),
        transport='same-node CPU memfd; microbatch-only upload to GPU')
    (destination.parent/(destination.name+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination, Path(__file__).with_name('cpu_logits_transport.py'))))
