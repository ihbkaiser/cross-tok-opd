"""Validate the exact trainer overrides with upstream parsing; no Ray/GPU init."""
import json
from pathlib import Path
import sys

import nemo_rl
from nemo_rl.algorithms.xtoken_off_policy_distillation import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
from omegaconf import OmegaConf

register_omegaconf_resolvers()
source = Path(nemo_rl.__file__).resolve().parent.parent
config = load_config(str(source/'examples/configs/xtoken_off_policy_distillation.yaml'))
config = parse_hydra_overrides(config, sys.argv[1:])
config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
assert config.distillation['max_num_steps'] == 3
assert config.policy['dtensor_cfg']['automodel_kwargs']['force_hf'] is True
assert config.policy['dtensor_cfg']['automodel_kwargs']['attn_implementation'] == 'sdpa'
assert config.cluster['gpus_per_node'] == 2
tokenizer = get_tokenizer(config.policy['tokenizer'])
dataset, validation = setup_response_data(tokenizer, config.data, env_configs=None)
assert len(dataset) >= 24
assert dataset[0] is not None
assert validation is None
print(json.dumps(dict(config_valid=True, data_valid=True, training_samples=len(dataset),
                      steps=3, gpu_count=2, backend='HF-SDPA')))
