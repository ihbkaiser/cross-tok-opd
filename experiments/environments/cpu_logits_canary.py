"""Two-process CUDA/CPU transport parity, no model weights or training run."""
import gc
import json
import os

import torch
import torch.distributed as dist
from nemo_rl.models.policy import vdt_cpu_logits as transport
from nemo_rl.models.policy.utils import ensure_teacher_ipc_buffer, get_handle_from_tensor, rebuild_cuda_tensor_from_ipc
from nemo_rl.algorithms.x_token.loss_utils import rebuild_teacher_full_logits_from_ipc

rank = int(os.environ['LOCAL_RANK'])
# Both processes use the same physical GPU for each round, matching producer /
# consumer CUDA IPC without requiring inter-GPU P2P, unavailable on this host.
dist.init_process_group('gloo')
shape = (2, 1, 17, 31)


def entries(handle):
    return [dict(teacher_shards=[dict(payload_ipc=handle, buf_idx=1, sample_index_in_buf=0,
        storage_shape=shape, actual_shape=(17, 31), tp_rank=0, cp_rank=0, tp_size=1, cp_size=1,
        world_rank=1-rank, vocab_start_index=0, vocab_end_index=31, global_seq_start=0,
        full_vocab_size=31, full_seq_len=17, vocab_sharded=False, sequence_sharded=False)])]


def loss(x, logits):
    return torch.nn.functional.kl_div(torch.log_softmax(x, -1), torch.softmax(logits, -1),
                                      reduction='batchmean') + .1 * x.square().mean()
for device in (0, 1):
    torch.cuda.set_device(device)
    reference = (torch.arange(2 * 17 * 31, device='cuda', dtype=torch.float32).reshape(shape) / 100).bfloat16().float()
    source = reference + rank
    source = source.bfloat16().float()  # synthetic exact BF16-origin fixture, not model quantization
    storage, cpu_handle = ensure_teacher_ipc_buffer(None, None, *shape, source.dtype, source.device)
    assert storage.device.type == 'cpu'
    storage.copy_(source)
    torch.cuda.synchronize()
    same, same_handle = ensure_teacher_ipc_buffer(storage, cpu_handle, *shape, source.dtype, source.device)
    assert same is storage and same_handle == cpu_handle
    rebuilt = rebuild_cuda_tensor_from_ipc(cpu_handle, device)
    actual = rebuilt[:] if getattr(transport, 'PACKED', False) else rebuilt
    assert torch.equal(actual.view(torch.int32), source.cpu().view(torch.int32))
    if getattr(transport, 'PACKED', False):
        special = torch.tensor([-0.0, -1.25, 0.0, 1e-30, 1e30], device='cuda').bfloat16().float()
        packed = torch.empty_like(special, device='cpu', dtype=torch.int16)
        transport.encode_into(packed, special)
        assert torch.equal(transport.decode(packed).view(torch.int32), special.cpu().view(torch.int32))
        bad = torch.full_like(source, 0.1)
        try:
            storage.copy_(bad)
        except ValueError as error:
            assert 'nonzero FP32 low bits' in str(error)
        else:
            raise AssertionError('Lossy source must be rejected, never rounded')
    handles = [None, None]
    dist.all_gather_object(handles, (cpu_handle, get_handle_from_tensor(source)))
    peer = handles[1-rank]
    ram = rebuild_teacher_full_logits_from_ipc(entries(peer[0]), None, device)
    cuda = rebuild_teacher_full_logits_from_ipc(entries(peer[1]), None, device)
    assert ram.is_cuda and ram.shape == (1, 17, 31)
    expected = (reference[1]+(1-rank)).bfloat16().float()
    assert torch.equal(ram.view(torch.int32), cuda.view(torch.int32))
    assert torch.equal(ram, expected)
    torch.manual_seed(42)
    student = torch.randn_like(ram, requires_grad=True)
    student2 = student.detach().clone().requires_grad_(True)
    loss1, loss2 = loss(student, ram), loss(student2, cuda)
    loss1.backward(); loss2.backward()
    assert torch.equal(loss1, loss2) and torch.equal(student.grad, student2.grad)
    assert torch.isfinite(student.grad).all()
    print(json.dumps(dict(rank=rank, gpu=device, cross_process_values_bitwise=True, loss_bitwise=True,
        gradient_bitwise=True, gpu_microbatch_shape=list(ram.shape),
        lossless_encoding=getattr(transport, 'PACKED', False),
        scope='same-GPU cross-process transport parity on each GPU; not full X-Token optimizer equivalence')), flush=True)
    del ram, cuda, loss1, loss2, student, student2
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    del same, storage, source, rebuilt, actual
    gc.collect()
    dist.barrier()
dist.destroy_process_group()
