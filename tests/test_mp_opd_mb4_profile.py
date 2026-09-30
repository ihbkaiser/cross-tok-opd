import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("mb4_import",Path(__file__).resolve().parents[1]/"experiments/modal/import_mp_opd_runai_logs.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_guarded_profile_rejects_bypass_and_wrong_source(tmp_path):
    launch={"source_commit":"f0c218183bb7043afa8e2b07c8313cac2ad1602e","source_dirty":"",
            "options":{"micro_train_batch_size":4,"train_batch_size":64,"diagnostic_max_updates":0,
                       "exact_token_trajectory":True,"diagnostic_collapse_gate":True,"use_wandb":False,"mp_opd_mode":"atomic"}}
    path=tmp_path/"launch-config.json";path.write_text(json.dumps(launch))
    e={"run_dir":str(tmp_path),"config":{"kd":{"mp_opd_mode":"atomic"},"train":{"micro_train_batch_size":4}},
       "bypass":[],"rows":[{"trajectory_logprob_abs_mean":.007,"trajectory_logprob_abs_p99":.12}]}
    result=m.import_profile(e,"mb4-guarded")
    assert result["bypass"] is False
    e["bypass"]=[{}]
    with pytest.raises(ValueError):m.import_profile(e,"mb4-guarded")
    e["bypass"]=[];launch["source_dirty"]=" M code.py";path.write_text(json.dumps(launch))
    with pytest.raises(ValueError):m.import_profile(e,"mb4-guarded")
