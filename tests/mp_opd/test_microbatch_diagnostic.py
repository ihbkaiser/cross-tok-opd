"""Execute the launcher guards without importing GPU/model dependencies."""
import ast
from pathlib import Path
import pytest

SOURCE = Path(__file__).parents[2]/'experiments/runai/run_single_gpu.py'


def validate(micro, meta, limit):
    tree = ast.parse(SOURCE.read_text())
    guards = [node for node in tree.body if isinstance(node, ast.If)
              and any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                      and n.value.startswith(('Student microbatch must', 'Meta microbatch must',
                                             'Microbatch overrides are'))
                      for n in ast.walk(node))]
    assert len(guards) == 3
    limit_guards = [node for node in tree.body if isinstance(node, ast.Assert)
                    and isinstance(node.test, ast.Compare)
                    and any(isinstance(n, ast.Name) and n.id == "limit"
                            for n in ast.walk(node.test))]
    assert len(limit_guards) == 1
    admission_assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                             and any(isinstance(target, ast.Name)
                                     and target.id in {"micro_recipe", "exact_soft_alternating_full"}
                                     for target in node.targets)]
    opts = {
        'micro_train_batch_size': micro,
        'mp_opd_meta_microbatch_size': meta,
        'kd_algorithm': 'mp_opd',
        'mp_opd_mode': 'soft',
        'mp_opd_alternating': True,
        'mp_opd_offload_adam_moments': True,
        'attn_implementation': 'eager',
        'train_batch_size': 64,
        'max_len': 4096,
        'rollout_batch_size': 64,
        'generate_max_len': 4096,
        'lr_scheduler_horizon_steps': 312,
        'exact_token_trajectory': True,
        'enforce_max_sequence_length': True,
    }
    exec(compile(ast.Module(body=admission_assignments + limit_guards + guards, type_ignores=[]),
                 str(SOURCE), 'exec'),
         dict(opts=opts, limit=limit, mode='soft'))


@pytest.mark.parametrize('micro,meta,limit', [(4,4,312),(64,16,10),(8,8,30),(1,1,2),(1,4,312),(1,4,0)])
def test_valid_recipe(micro, meta, limit):
    validate(micro,meta,limit)


@pytest.mark.parametrize('micro,meta,limit', [(3,4,10),(4,3,10),(8,4,312),(4,8,0),(1,4,31)])
def test_invalid_or_unqualified_production_override(micro, meta, limit):
    with pytest.raises(ValueError): validate(micro,meta,limit)


def test_update_limit_remains_hard_cap():
    with pytest.raises(AssertionError):
        validate(4, 4, 313)
