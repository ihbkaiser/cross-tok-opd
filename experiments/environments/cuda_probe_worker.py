"""Diagnostic extension only: compare CUDA buffers across Ray init/method threads."""
import json
import os
import threading

import ray
import torch
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2Impl


@ray.remote
class BufferProbeWorker(DTensorPolicyWorkerV2Impl):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inspect_buffers('constructor_complete')

    def inspect_buffers(self, stage):
        print(json.dumps(dict(stage=stage, rank=self.rank, thread=threading.get_ident(),
                              device=torch.cuda.current_device(),
                              visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                              launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'))), flush=True)
        torch.cuda.synchronize()
        for name, v in self.model.named_buffers():
            print(json.dumps(dict(stage=stage+'_buffer', rank=self.rank, name=name,
                                  shape=list(v.shape), device=str(v.device),
                                  dtype=str(v.dtype))), flush=True)
            cpu = v.to('cpu')
            print(json.dumps(dict(stage=stage+'_copy_ok', rank=self.rank, name=name,
                                  finite=bool(torch.isfinite(cpu).all()))), flush=True)
