"""Regression tests for compatibility with the pre-recovery native W&B schema."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/evaluation'))
from align_campaign_wandb_schema import native_payload,missing_rows,tags_for,CHECKPOINT_COLUMNS

def point():
    return {'step':40,'benchmark':'gsm8k','eval_scores':{42:.1,43:.2,44:.3},'mean':.2,'eval_std':.1,'eval_n':3}

def test_native_columns_and_axes_match_legacy():
    rows,cp,dp=native_payload([point()])
    assert CHECKPOINT_COLUMNS==['checkpoint_step','benchmark','mean','sample_std','seed42','seed43','seed44']
    assert cp==[[40,'gsm8k',.2,.1,.1,.2,.3]]
    assert rows[40]['eval/curve/gsm8k']==.2
    assert rows[40]['eval/dispersion_step']==40
    assert rows[40]['eval/dispersion/gsm8k/seed43']==.2
    assert missing_rows(rows,list(rows.values()))==[]

def test_native_does_not_overwrite_protocol_conflict():
    rows,_,_=native_payload([point()]);old=[{'eval/checkpoint_step':40,'eval/curve/gsm8k':.8}]
    with pytest.raises(ValueError,match='protocol/value conflict'):missing_rows(rows,old)

def test_partial_native_table_keeps_seed_but_no_mean():
    p=point();p.update(eval_scores={42:.1},mean=None,eval_std=None,eval_n=1)
    rows,cp,dp=native_payload([p]);assert 'eval/curve/gsm8k' not in rows[40]
    assert cp==[[40,'gsm8k',None,None,.1,None,None]]

def test_tags_are_same_dimensions_across_training_seeds():
    r={'method':'fixed','target':312,'primary':True,'train_seed':42,'eval_cells':12}
    a=tags_for(r,[]);r['train_seed']=43;b=tags_for(r,[])
    assert set(a)-set(b)=={'train-seed:42'}
    assert set(b)-set(a)=={'train-seed:43'}
    assert 'method:mp-opd' in a and 'variant:fixed2' in a and 'schema:experiment-v1' in a
