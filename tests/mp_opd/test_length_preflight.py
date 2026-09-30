import pytest
from experiments.mp_opd.real_oracle import preflight_lengths


class Tokenizer:
    eos_token_id=1
    def apply_chat_template(self,messages,**kwargs):return [2]*int(messages[0]['content'])
    def encode(self,text,**kwargs):return [3]*int(text)


def test_select_long_prompt_fails_before_gpu():
    short={'messages':[{'content':'10'}],'reference':'5'}
    long={'id':'too-long','messages':[{'content':'1025'}],'reference':'5'}
    with pytest.raises(ValueError,match='role=select.*prompt=1025'):
        preflight_lengths([{'rollout':short,'select':[short,long],'eval':short}],Tokenizer(),Tokenizer(),1024,2048)


def test_explicit_larger_cap_preserves_full_prompt():
    row={'messages':[{'content':'1500'}],'reference':'50'}
    result=preflight_lengths([dict(rollout=row,select=row,eval=row)],Tokenizer(),Tokenizer(),4096,2048)
    assert result=={'student_prompt':1500,'teacher_prompt':1500,'reference':51}


def test_oracle_retry_preserves_deadline_budget():
    from experiments.runai.recover_oracle import budgets
    primary, secondary = budgets(20000, 10000)
    assert primary < 10000
    assert 2 * secondary < primary
    with pytest.raises(ValueError, match="clock"):
        budgets(11000, 10000)


def test_reference_cap_counts_eos():
    row={'messages':[{'content':'10'}],'reference':'4096'}
    with pytest.raises(ValueError):
        preflight_lengths([dict(rollout=row,select=row,eval=row)],Tokenizer(),Tokenizer(),4096,4096)
