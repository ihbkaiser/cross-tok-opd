import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kdflow.training_checkpoint import (begin, publish, inspect, write_payload,
    snapshot_training, restore_training, capture_rng, restore_rng, seeded_sampling)
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("sampler", Path(__file__).parents[1]/"kdflow/utils/distributed_sampler.py")
sampler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler_module)
DistributedSampler = sampler_module.DistributedSampler


def components():
    model = torch.nn.Linear(3, 2).double()
    projector = torch.nn.Linear(2, 2).double()
    energy = torch.nn.Linear(2, 1).double()
    opt = torch.optim.AdamW([*model.parameters(), *projector.parameters()], lr=.003)
    eo = torch.optim.AdamW(energy.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=.8)
    algo = SimpleNamespace(student=model, projector=projector, energy=energy, energy_optimizer=eo)
    return model, algo, opt, scheduler


def update(c, index):
    m, a, o, s = c
    x = torch.randn(4, 3, dtype=torch.double) * (random.random() + np.random.rand())
    if index % 4 == 0:
        a.energy(m(x).detach()).square().mean().backward()
        a.energy_optimizer.step(); a.energy_optimizer.zero_grad(set_to_none=True)
    a.projector(m(x)).square().mean().backward()
    torch.nn.utils.clip_grad_norm_([*m.parameters(), *a.projector.parameters()], .3)
    o.step(); o.zero_grad(set_to_none=True); s.step()


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a,b): equal(x,y)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a,b)
    else: assert a == b


@pytest.mark.parametrize("pause", [1, 3, 4, 5])
def test_adam_projector_energy_scheduler_rng_resume(pause):
    torch.manual_seed(12); random.seed(12); np.random.seed(12)
    continuous = components()
    for i in range(1,9): update(continuous,i)
    expected = snapshot_training(*continuous, accumulation_step=0)
    torch.manual_seed(12); random.seed(12); np.random.seed(12)
    c = components()
    for i in range(1, pause+1): update(c,i)
    payload = copy.deepcopy(snapshot_training(*c, accumulation_step=0))
    recovered = components()
    restore_training(payload, *recovered)
    for i in range(pause+1, 9): update(recovered,i)
    equal(expected, snapshot_training(*recovered, accumulation_step=0))


def test_transaction_old_latest_survives_partial_and_corruption(tmp_path):
    c = components(); update(c,1)
    directory = begin(tmp_path, 1)
    info = write_payload(directory/'rank0.pt', snapshot_training(*c, accumulation_step=0))
    publish(tmp_path,directory,step=1,world_size=1,contract={'seed':42},client={'cursor':64},rank_files=[info])
    partial = begin(tmp_path,2)
    write_payload(partial/'rank0.pt', {'incomplete': True})
    assert inspect(tmp_path,{'seed':42},1)[0] == directory
    with pytest.raises(ValueError,match='contract'):
        inspect(tmp_path,{'seed':43},1)
    with (directory/'rank0.pt').open('ab') as f: f.write(b'bad')
    with pytest.raises(ValueError,match='checksum'):
        inspect(tmp_path,{'seed':42},1)


def test_pending_gradient_rejected():
    c=components(); c[0](torch.ones(1,3,dtype=torch.double)).sum().backward()
    with pytest.raises(ValueError,match='boundary'):
        snapshot_training(*c,accumulation_step=0)


def test_disabled_cadence_strict_json_contract():
    from kdflow.training_checkpoint import normalize_config
    value=normalize_config({'eval_steps':float('inf'),'betas':(.9,.98)})
    assert json.loads(json.dumps(value,allow_nan=False))==value
    assert value['eval_steps']=='Infinity'
    with pytest.raises(ValueError,match='NaN'):
        normalize_config({'learning_rate':float('nan')})


def test_sampler_and_stateless_rollout_seed_restart():
    sampler=DistributedSampler(list(range(10000)),1,0,True,42,True)
    sampler.set_epoch(1); original=list(sampler)
    sampler.set_epoch(1,consumed_samples=64*40)
    assert list(sampler)==original[64*40:]
    before=capture_rng()
    x=seeded_sampling({'temperature':.6},seed=42,step=41,count=64)
    equal(before,capture_rng())
    assert x==seeded_sampling({'temperature':.6},seed=42,step=41,count=64)
    assert x!=seeded_sampling({'temperature':.6},seed=42,step=42,count=64)


def test_retention_preserves_partial_evidence(tmp_path):
    from kdflow.training_checkpoint import prune_complete
    folders=[]
    for step in range(4):
        d=begin(tmp_path,step); folders.append(d)
        info=write_payload(d/'rank0.pt', {'step':step})
        publish(tmp_path,d,step=step,world_size=1,contract={},client={},rank_files=[info])
    partial=begin(tmp_path,5)
    (partial/'rank0.pt').write_bytes(b'incomplete')
    prune_complete(tmp_path)
    assert not folders[0].exists() and not folders[1].exists()
    assert folders[2].exists() and folders[3].exists() and partial.exists()
