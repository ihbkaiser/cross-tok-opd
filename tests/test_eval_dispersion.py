import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/evaluation"))
from publish_wandb_uncertainty import stats,values,BENCHES

def test_sample_variance():
    z=stats([0,.5,1])
    assert z["variance"]==.25 and z["std"]==.5
    assert z["min"]==0 and z["max"]==1

def test_macro_uses_per_seed_covariance():
    cp={"benchmarks":{b:{"scores":([0,.5,1] if i%2==0 else [1,.5,0])} for i,b in enumerate(BENCHES)}}
    assert stats(values(cp,"macro"))["variance"]==0
