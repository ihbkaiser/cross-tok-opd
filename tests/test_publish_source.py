import importlib.util
from pathlib import Path
import subprocess

spec=importlib.util.spec_from_file_location("publisher",Path(__file__).resolve().parents[1]/"experiments/runai/publish_source.py")
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)


def test_preserves_other_hf_assets(tmp_path):
    bundle=tmp_path/"simct-b200-portable.bundle";bundle.write_bytes(b"test")
    old={"eval_queue":{"helper":"pinned"},"source_commit":"old","abc_transfer":{"x":"y"}}
    new=P.manifest_update(old,"new",bundle)
    assert old['source_commit']=='old'
    assert new['eval_queue']==old['eval_queue']
    assert new['abc_transfer']==old['abc_transfer']
    assert new['bundle_bytes']==4 and new['bundle_sha256']==P.sha(bundle)


def test_bundle_clone_preserves_commit(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    P.git(repo,'init','-b',P.BRANCH)
    (repo/'code.py').write_text('print(1)\n')
    P.git(repo,'add','code.py')
    P.git(repo,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','test')
    head=P.git(repo,'rev-parse','HEAD');stage=tmp_path/'stage';stage.mkdir()
    bundle=P.bundle_build(repo,stage,head)
    dest=tmp_path/'clone'
    subprocess.run(['git','clone','-b',P.BRANCH,str(bundle),str(dest)],check=True,capture_output=True)
    assert P.git(dest,'rev-parse','HEAD')==head
    assert (dest/'code.py').read_bytes()==(repo/'code.py').read_bytes()
