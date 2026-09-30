"""CPU-only exact target/config/token-length gate before any training attempt."""
import argparse
import json
from pathlib import Path

import nemo_rl
from nemo_rl.algorithms.xtoken_off_policy_distillation import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.data.cross_tokenizer_collate import CrossTokenizerCollator
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument('--expected-steps', type=int, choices=(3, 10), required=True)
args, settings = parser.parse_known_args()
minimum_presentations = args.expected_steps * 64
register_omegaconf_resolvers()
source = Path(nemo_rl.__file__).resolve().parent.parent
config = load_config(str(source/'examples/configs/xtoken_off_policy_distillation.yaml'))
config = MasterConfig(**OmegaConf.to_container(parse_hydra_overrides(config, settings), resolve=True))
assert config.distillation['max_num_steps'] == args.expected_steps
assert config.distillation['num_prompts_per_step'] == 64
assert config.cluster['gpus_per_node'] == 2
assert not config.loss_fn['gold_loss'] and not config.loss_fn['xtoken_loss']
assert config.loss_fn['ce_loss_scale'] == 0.1
assert not config.checkpointing['enabled']
teacher = config.teachers[0].policy_config()
for policy in (config.policy, teacher):
    assert policy['train_global_batch_size'] == 64 and policy['train_micro_batch_size'] == 1
    assert policy['max_total_sequence_length'] == 2048 and policy['precision'] == 'bfloat16'
    assert policy['dtensor_cfg']['automodel_kwargs'] == dict(force_hf=True, attn_implementation='sdpa')
assert config.data['max_input_seq_length'] == 2048
assert config.data['train']['characters_per_sample'] == 16384
tokenizer = get_tokenizer(config.policy['tokenizer'])
teacher_tokenizer = get_tokenizer(teacher['tokenizer'])
dataset, validation = setup_response_data(tokenizer, config.data, env_configs=None)
assert validation is None and len(dataset) >= minimum_presentations
length_reports = []
for tok in (tokenizer, teacher_tokenizer):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    lengths = []
    for start in range(0, len(dataset), 32):
        texts = [dataset[i]['message_log'][0]['content'] for i in range(start, min(start+32, len(dataset)))]
        ids, mask = CrossTokenizerCollator._tokenize_batch(texts, tok, 2048, 1)
        assert ids.shape[1] == 2048
        lengths.extend(mask.sum(1).tolist())
    assert sum(n == 2048 for n in lengths) >= minimum_presentations, 'Insufficient full-length examples'
    length_reports.append(dict(samples=len(lengths), min=min(lengths), max=max(lengths),
        mean=sum(lengths)/len(lengths), full_length_samples=sum(n == 2048 for n in lengths)))
out = Path(config.logger['log_dir']).parent
(out/'config-resolved.json').write_text(json.dumps(config.model_dump(), indent=2, default=str) + '\n')
report = dict(config_valid=True, samples=len(dataset), lengths=length_reports,
              global_batch=64, sequence_length=2048, steps=args.expected_steps, gpu_count=2,
              scope='full-projection P-KL+CE off-policy workload only')
(out/'data-config-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
