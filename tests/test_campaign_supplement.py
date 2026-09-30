import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/evaluation'))
from supplement_campaign_training import missing_training

def test_supplement_only_missing_fields():
    rows=[{'step':1,'loss':.2,'lr':.01}]
    assert missing_training(rows,[{'train/global_step':1,'train/loss':.2}])==[{'train/global_step':1,'train/lr':.01}]
    assert missing_training(rows,[{'train/global_step':1,'train/loss':.2,'train/lr':.01}])==[]

def test_supplement_refuses_conflicting_history():
    with pytest.raises(ValueError,match='source training conflict'):
        missing_training([{'step':1,'loss':.2}],[{'train/global_step':1,'train/loss':.3}])