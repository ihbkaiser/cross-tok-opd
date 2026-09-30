"""Lossless FP32 wire encoding for BF16-origin logits; never quantizes input.

Every source value must already have zero low 16 bits in its FP32 encoding.
Otherwise fail before training: do NOT round, cast values, truncate or fall back.
Logical tensors and all consumer math stay FP32. Only redundant zero bits are
omitted in cgroup-accounted shared RAM; selected slices reconstruct exact bits.
"""
import fcntl
import math
import os
import sys
import uuid
import weakref

import torch

TAG = 'vdt-lossless-fp32-hi16-memfd-v1'
PACKED = True
MAX_BYTES = 20*1024**3
CHECK_CHUNK_BYTES = 16*1024**2


def seal_size(fd):
    assert sys.platform == 'linux' and sys.byteorder == 'little'
    add, get = getattr(fcntl, 'F_ADD_SEALS', 1033), getattr(fcntl, 'F_GET_SEALS', 1034)
    mask = getattr(fcntl, 'F_SEAL_SHRINK', 2) | getattr(fcntl, 'F_SEAL_GROW', 4)
    fcntl.fcntl(fd, add, mask)
    assert fcntl.fcntl(fd, get) & mask == mask


def encode_into(destination, source):
    assert source.dtype == torch.float32 and source.shape == destination.shape
    source = source.contiguous()
    width = source.shape[-1]
    source_rows = source.reshape(-1, width)
    destination_rows = destination.reshape(-1, width)
    rows_per_chunk = max(1, CHECK_CHUNK_BYTES // (width * source.element_size()))
    for start in range(0, source_rows.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, source_rows.shape[0])
        source_chunk = source_rows[start:stop]
        words = source_chunk.view(torch.int16)
        # Reinterpret bits, not a numeric conversion to lower precision.  The
        # bounded slabs avoid multi-GiB reduction workspaces on 24 GiB GPUs.
        if torch.count_nonzero(words[..., 0::2]).item() != 0:
            raise ValueError('Lossless encoding rejected: nonzero FP32 low bits; no rounding allowed')
        assert torch.isfinite(source_chunk).all().item(), 'Non-finite teacher logits'
        destination_rows[start:stop].copy_(words[..., 1::2])


def decode(source):
    return source.to(torch.int32).bitwise_left_shift(16).view(torch.float32)


class Destination:
    def __init__(self, tensor):
        self.tensor = tensor

    def copy_(self, source):
        encode_into(self.tensor, source)
        return self


class LogicalBuffer:
    def __init__(self, storage, producer):
        self.storage, self.producer = storage, producer
        self.shape, self.device, self.dtype = storage.shape, storage.device, torch.float32

    def detach(self):
        return self

    def __getitem__(self, index):
        view = self.storage[index]
        return Destination(view) if self.producer else decode(view)

    def copy_(self, source):
        assert self.producer
        encode_into(self.storage, source)
        return self


def ensure_cpu_buffer(storage, handle, shape, dtype):
    assert dtype == torch.float32, 'Packed protocol is exact FP32 only'
    shape = tuple(int(d) for d in shape)
    assert len(shape) == 4 and all(d > 0 for d in shape)
    nbytes = math.prod(shape)*2
    assert nbytes <= MAX_BYTES
    if storage is not None:
        assert isinstance(storage, LogicalBuffer) and storage.producer
        assert tuple(storage.shape) == shape and handle[0] == TAG
        return storage, handle
    fd = os.memfd_create('vdt-teacher-logits-'+uuid.uuid4().hex, os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
    try:
        os.fchmod(fd, 0o600)
        os.ftruncate(fd, nbytes)
        seal_size(fd)
        raw = torch.from_file(f'/proc/{os.getpid()}/fd/{fd}', shared=True,
                              size=math.prod(shape), dtype=torch.int16).reshape(shape)
        stat = os.fstat(fd)
        handle = (TAG, os.getpid(), fd, shape, stat.st_dev, stat.st_ino, nbytes)
        weakref.finalize(raw, os.close, fd)
        return LogicalBuffer(raw, True), handle
    except BaseException:
        os.close(fd)
        raise


def rebuild_cpu_buffer(handle):
    tag,pid,fd,shape,device,inode,nbytes = handle
    assert tag == TAG and type(pid) is int and pid > 0 and type(fd) is int and fd >= 0
    assert len(shape) == 4 and all(type(d) is int and d > 0 for d in shape)
    assert math.prod(shape)*2 == nbytes <= MAX_BYTES
    path = f'/proc/{pid}/fd/{fd}'
    assert os.readlink(path).startswith('/memfd:vdt-teacher-logits-')
    stat = os.stat(path)
    assert (stat.st_dev,stat.st_ino,stat.st_size) == (device,inode,nbytes)
    raw = torch.from_file(path, shared=True, size=math.prod(shape), dtype=torch.int16).reshape(shape)
    return LogicalBuffer(raw, False)


def transport_view(view, device):
    assert view.dtype == torch.float32
    return view.to(device=device) if view.device.type == 'cpu' else view
