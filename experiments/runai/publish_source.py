#!/usr/bin/env python3
"""Publish one committed source revision to authorized GitHub and HF bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

BRANCH="vdt/ops/b200-portable"
ORIGIN="https://github.com/sontungkieu/SimCT.git"
REPO="codemaivanngu/simct"


def git(root,*args):
    return subprocess.check_output(["git",*args],cwd=root,text=True).strip()


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_update(previous,head,bundle):
    result=dict(previous)
    result.update(source_branch=BRANCH,source_commit=head,bundle_file=bundle.name,
                  bundle_sha256=sha(bundle),bundle_bytes=bundle.stat().st_size,
                  history="complete",format="git-bundle")
    return result


def bundle_build(root,stage,head):
    bundle=stage/"simct-b200-portable.bundle"
    git(root,"bundle","create",str(bundle),BRANCH)
    git(root,"bundle","verify",str(bundle))
    refs=git(root,"bundle","list-heads",str(bundle)).splitlines()
    if refs!=[head+" refs/heads/"+BRANCH]: raise ValueError("Bundle ref mismatch")
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publish",action="store_true",help="Push GitHub then publish HF; default checks local state only")
    p.add_argument("--job-manager-source",type=Path,help="Include a clean, GitHub-verified job-manager source archive")
    args=p.parse_args()
    root=Path(git(Path(__file__).resolve().parent,"rev-parse","--show-toplevel"))
    if git(root,"branch","--show-current")!=BRANCH: raise ValueError("Unexpected branch")
    if git(root,"remote","get-url","--push","origin")!=ORIGIN: raise ValueError("Unexpected GitHub target")
    if git(root,"status","--porcelain","--untracked-files=no"): raise ValueError("Tracked edits must be committed first")
    if git(root,"rev-parse","--is-shallow-repository")!="false": raise ValueError("Full history required")
    head=git(root,"rev-parse","HEAD")
    print("SOURCE_COMMIT="+head,flush=True)
    if not args.publish:
        print("CHECK_ONLY: no push, HF write, or GPU launch. Untracked files are not published.");return
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    token=os.environ.get("HF_TOKEN")
    if not token: raise ValueError("Provide HF_TOKEN through the existing secret environment")
    api=HfApi(token=token)
    parent=api.repo_info(REPO).sha
    artifact=root/"remote_artifacts";artifact.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="publish-"+head[:8]+"-",dir=artifact))
    receipt={"source_commit":head,"branch":BRANCH,"github":ORIGIN,"hf_repo":REPO,
             "hf_parent":parent,"github_verified":False,"hf_verified":False}
    def save():
        temp=stage/"receipt.tmp";temp.write_text(json.dumps(receipt,indent=2)+"\n");temp.replace(stage/"receipt.json")
    save()
    previous_path=hf_hub_download(REPO,"source-manifest.json",revision=parent,token=token,cache_dir=str(stage/"cache"))
    previous=json.loads(Path(previous_path).read_text())
    bundle=bundle_build(root,stage,head)
    manifest=manifest_update(previous,head,bundle)
    extra=[]
    if args.job_manager_source:
        jm=args.job_manager_source.resolve()
        if git(jm,'status','--porcelain'):raise ValueError('Dirty job-manager source')
        jm_head=git(jm,'rev-parse','HEAD')
        if git(jm,'remote','get-url','origin')!='https://github.com/sontungkieu/job-manager.git':raise ValueError('Unexpected job-manager origin')
        if git(jm,'ls-remote','origin','refs/heads/main')!=jm_head+'\trefs/heads/main':raise ValueError('job-manager GitHub ref not verified')
        archive=stage/'job-manager-source.tar'
        git(jm,'archive','--format=tar','--prefix=job-manager/','--output',str(archive),jm_head)
        manifest['job_manager']={'file':archive.name,'sha256':sha(archive),'source_commit':jm_head,'github':'https://github.com/sontungkieu/job-manager'}
        extra.append(archive)
    mp=stage/"source-manifest.json";mp.write_text(json.dumps(manifest,indent=2)+"\n")
    def github_verify():
        if git(root,"ls-remote","origin","refs/heads/"+BRANCH)!=head+"\trefs/heads/"+BRANCH:
            raise ValueError("GitHub ref differs; stop and reconcile")
    git(root,"push","origin",head+":refs/heads/"+BRANCH)
    github_verify();receipt['github_verified']=True;save()
    if previous.get('source_commit')==head and previous.get('bundle_sha256')==manifest['bundle_sha256'] and previous.get('job_manager')==manifest.get('job_manager'):
        revision=parent
    else:
        # Optimistic concurrency: never overwrite an intervening HF publication.
        receipt['hf_submission']='pending';save()
        result=api.create_commit(repo_id=REPO,parent_commit=parent,
            operations=[CommitOperationAdd(path_in_repo=x.name,path_or_fileobj=str(x)) for x in (bundle,mp,*extra)],
            commit_message="Sync GitHub source "+head)
        revision=result.oid
    receipt['hf_revision']=revision;save()
    # Read immutable HF revision back anonymously, including the actual bundle.
    for local in (bundle,mp,*extra):
        downloaded=Path(hf_hub_download(REPO,local.name,revision=revision,token=False,cache_dir=str(stage/'verify-cache')))
        if local==mp:
            actual=json.loads(downloaded.read_text())
            for key in ('source_commit','bundle_sha256','bundle_bytes','source_branch','job_manager'):
                if key not in manifest:continue
                if actual[key]!=manifest[key]: raise ValueError('HF manifest mismatch')
        elif sha(downloaded)!=sha(local): raise ValueError('HF bundle readback mismatch')
    github_verify();receipt['hf_verified']=True;receipt['hf_submission']='verified';save()
    print('DUAL_PUBLISH_PASS '+json.dumps(receipt),flush=True)
    print('RECEIPT='+str(stage/'receipt.json'))

if __name__=='__main__': main()
