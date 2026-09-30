import random
import pytest
from kdflow.meta_data import MetaSampler, prompt_id


def test_disjoint_replay_and_shared_schedule_without_global_rng():
    prompts=[f'question {i}' for i in range(100)]
    rows=[dict(prompt=p,reference='reference '+p) for p in prompts]
    a=MetaSampler(rows,prompts,42,16)
    before=random.getstate()
    selected=a.batch(4,prompts[:64])
    assert len(selected)==16 and len({r['id'] for r in selected})==16
    assert not {r['id'] for r in selected} & {prompt_id(p) for p in prompts[:64]}
    assert MetaSampler(rows,prompts,42,16).batch(4,prompts[:64])==selected
    assert a.batch(8,prompts[:64])!=selected
    assert before==random.getstate()


def test_reference_membership():
    with pytest.raises(ValueError,match='outside'):
        MetaSampler([dict(prompt='test',reference='answer')],['train'],42,1)


@pytest.mark.parametrize('left,right',[
    ('train',' TRAIN '),('evaluate X','evaluate x'),('a\n  b','a b'),
])
def test_normalization_collision_preserves_original_pair_and_excludes_group(left,right):
    rows=[dict(prompt=left,reference='answer a'),dict(prompt=right,reference='answer b')]
    sampler=MetaSampler(rows,[left,right],42,1)
    replay=MetaSampler(rows[::-1]+rows,[right,left],42,1)
    assert sampler.audit['multiple_prompt_form_groups']==1
    observed=set()
    for step in range(30):
        batch=sampler.batch(step,[])
        assert batch==replay.batch(step,[])
        observed.add((batch[0]['prompt'],batch[0]['reference']))
    assert observed=={(left,'answer a'),(right,'answer b')}
    for excluded in (left,right):
        with pytest.raises(ValueError,match='disjoint'):sampler.batch(1,[excluded])


def test_multi_reference_is_order_and_duplicate_invariant():
    rows=[dict(prompt=f'q{i}',reference=f'a{i}') for i in range(100)]
    rows+=[dict(prompt='q1',reference='alternate answer')]
    before=random.getstate()
    a=MetaSampler(rows,[f'q{i}' for i in range(100)],42)
    b=MetaSampler(list(reversed(rows))+rows,[f'q{i}' for i in range(100)],42)
    for step in (1,4,156,312):
        assert a.batch(step,['q0'])==b.batch(step,['q0'])
        assert len({r['id'] for r in a.batch(step,['q0'])})==16
    assert a.audit['multi_reference_prompts']==1
    assert a.audit['groups_sha256']==b.audit['groups_sha256']
    assert random.getstate()==before


def test_reference_draw_does_not_bias_prompt_selection():
    rows=[dict(prompt=f'q{i}',reference='a') for i in range(20)]
    prompts=[r['prompt'] for r in rows]
    a=MetaSampler(rows,prompts,42)
    b=MetaSampler(rows+[dict(prompt='q0',reference='b')],prompts,42)
    for step in range(20):
        assert [r['id'] for r in a.batch(step,[]) ]==[r['id'] for r in b.batch(step,[])]
    one=MetaSampler([dict(prompt='q',reference=r) for r in ('a','b')],['q'],42,1)
    assert {one.batch(s,[])[0]['reference'] for s in range(30)}=={'a','b'}


def test_prepare_meta_filters_each_reference_and_rejects_empty():
    from types import SimpleNamespace
    from kdflow.meta_data import prepare_meta
    tok=SimpleNamespace(encode=lambda text,**kw:list(text))
    rows,excluded=prepare_meta([{'prompt':'q','label':'a'},{'prompt':'q','label':'long'}],
        lambda r:r['prompt'],tok,3)
    assert rows==[dict(prompt='q',reference='a')] and excluded==1
    with pytest.raises(ValueError,match='nonempty'):
        prepare_meta([{'label':''}],lambda r:'q',tok,3)
