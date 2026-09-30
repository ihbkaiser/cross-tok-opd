"""Same-node, full-precision teacher-logit transport in cgroup-accounted RAM.

memfd mappings share physical pages across processes without a disk artifact or
Ray object-store copy. No dtype/vocabulary reduction or optimizer changes.
Producer lifetime owns the descriptor; a weak finalizer closes it on release.
Consumer maps CPU storage and copies only the requested microbatch to CUDA.
"""
import fcntl
import math
import os
import sys
import uuid
import weakref

import torch

TAG = 'vdt-cpu-memfd-logits-v1'
DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
MAX_BYTES = 40 * 1024**3


def seal_size(fd):
    # Linux UAPI constants, verified against this host's linux/fcntl.h.
    # Some standalone CPython builds omit the exported names despite kernel support.
    assert sys.platform == 'linux'
    add = getattr(fcntl, 'F_ADD_SEALS', 1033)
    get = getattr(fcntl, 'F_GET_SEALS', 1034)
    mask = getattr(fcntl, 'F_SEAL_SHRINK', 2) | getattr(fcntl, 'F_SEAL_GROW', 4)
    fcntl.fcntl(fd, add, mask)
    assert fcntl.fcntl(fd, get) & mask == mask


def ensure_cpu_buffer(storage, handle, shape, dtype):
    shape = tuple(int(d) for d in shape)
    assert len(shape) == 4 and all(d > 0 for d in shape)
    dtype_name = str(dtype).removeprefix('torch.')
    assert dtype_name in DTYPES
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    assert nbytes <= MAX_BYTES, 'Teacher CPU buffer exceeds declared per-rank cap'
    if storage is not None:
        assert storage.device.type == 'cpu' and storage.dtype == dtype
        assert tuple(storage.shape) == shape, 'Shape change requires a new explicit attempt'
        assert handle[0] == TAG
        return storage, handle
    fd = os.memfd_create('vdt-teacher-logits-' + uuid.uuid4().hex,
                         os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.fchmod(fd, 0o600)
        os.ftruncate(fd, nbytes)
        seal_size(fd)
        path = f'/proc/{os.getpid()}/fd/{fd}'
        storage = torch.from_file(path, shared=True, size=math.prod(shape), dtype=dtype).reshape(shape)
        stat = os.fstat(fd)
        handle = (TAG, os.getpid(), fd, shape, dtype_name, stat.st_dev, stat.st_ino, nbytes)
        weakref.finalize(storage, os.close, fd)
        return storage, handle
    except BaseException:
        os.close(fd)
        raise


def rebuild_cpu_buffer(handle):
    tag, pid, fd, shape, dtype_name, device, inode, nbytes = handle
    assert tag == TAG and dtype_name in DTYPES
    assert type(pid) is int and pid > 0 and type(fd) is int and fd >= 0
    assert len(shape) == 4 and all(type(d) is int and d > 0 for d in shape)
    dtype = DTYPES[dtype_name]
    assert math.prod(shape) * torch.empty((), dtype=dtype).element_size() == nbytes <= MAX_BYTES
    path = f'/proc/{pid}/fd/{fd}'
    assert os.readlink(path).startswith('/memfd:vdt-teacher-logits-'), 'Unexpected producer handle'
    stat = os.stat(path)
    assert (stat.st_dev, stat.st_ino, stat.st_size) == (device, inode, nbytes), 'Stale producer handle'
    return torch.from_file(path, shared=True, size=math.prod(shape), dtype=dtype).reshape(shape)


def transport_view(view, device):
    """Retain CUDA zero-copy path; CPU path uploads the already sliced tensor."""
    return view.to(device=device) if view.device.type == 'cpu' else view
