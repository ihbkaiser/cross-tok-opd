"""CPU-only inspection/export of pinned B200 runtime."""
import modal,subprocess,json,os,hashlib,shlex,tarfile,shutil
from pathlib import Path
IMAGE='docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
app=modal.App('simct-b200-portable-inspect')
image=modal.Image.from_registry(IMAGE).entrypoint([]).add_local_file('/home/tung/simct-data-evidence/b200-source.tar','/payload/source.tar')
@app.function(image=image,cpu=1,memory=2048,timeout=180,retries=0)
def inspect():
    code="import sys,json;print(json.dumps(dict(executable=sys.executable,base_prefix=sys.base_prefix,prefix=sys.prefix)))"
    for cmd in [['/opt/venvs/simct-b200/bin/python','-c',code],['cat','/etc/os-release'],['readlink','-f','/opt/venvs/simct-b200/bin/python'],['du','-sh','/opt/venvs/simct-b200','/usr/local/cuda'],['ls','-ld','/usr/local/cuda'],['ldd','/opt/venvs/simct-b200/bin/python']]:
        r=subprocess.run(cmd,text=True,capture_output=True);print(cmd,r.stdout,r.stderr,flush=True)

REPO='codemaivanngu/simct-b200-portable-cu130'
volume=modal.Volume.from_name('simct-b200-portable-export',create_if_missing=True)
@app.function(image=image,cpu=4,memory=8192,timeout=1800,retries=0,volumes={'/export':volume})
def package(source_commit:str):
    root=Path('/export/release-v1');root.mkdir(exist_ok=True)
    if (root/'receipt.json').exists():raise RuntimeError('Already uploaded; inspect receipt')
    python='/opt/venvs/simct-b200/bin/python'
    base=Path(subprocess.check_output([python,'-c','import sys;print(sys.base_prefix)'],text=True).strip())
    cuda=Path('/usr/local/cuda').resolve()
    assert str(base).startswith('/opt/python/') and str(cuda).startswith('/usr/local/cuda-')
    support=Path('/opt/simct-portable-libs');support.mkdir(exist_ok=True)
    listing=subprocess.check_output(['ldconfig','-p'],text=True)
    names={'libnuma.so.1','libstdc++.so.6','libgcc_s.so.1','libgomp.so.1','libaio.so.1','libibverbs.so.1','librdmacm.so.1','libnl-3.so.200','libnl-route-3.so.200'}
    for line in listing.splitlines():
        fields=line.split()
        if fields and fields[0] in names:
            src=Path(fields[-1])
            if src.is_file():shutil.copy2(src,support/fields[0])
    targets=[Path('/opt/venvs/simct-b200'),base,cuda,support]
    (root/'targets.txt').write_text(''.join(str(x)+'\n' for x in targets))
    print('ARCHIVE_TARGETS',[(str(x),subprocess.check_output(['du','-sh',str(x)],text=True).split()[0]) for x in targets],flush=True)
    archive=root/'runtime.tar.gz'
    subprocess.run(['tar','-I','gzip -1','-cf',str(archive),'-C','/','--exclude=__pycache__','--exclude=*.pyc',*[str(x).lstrip('/') for x in targets]],check=True,timeout=900)
    with tarfile.open(archive,'r:gz') as tf:
        prefixes=tuple(str(x).lstrip('/') for x in targets)
        for m in tf:
            if not any(m.name==q or m.name.startswith(q+'/') for q in prefixes):raise ValueError('Archive outside allowlist')
    shutil.copyfile('/payload/source.tar',root/'source.tar')
    packages=subprocess.check_output([python,'-c','import importlib.metadata as m,json;print(json.dumps({x.metadata["Name"]:x.version for x in m.distributions()},indent=2))'],text=True)
    (root/'packages.json').write_text(packages)
    run_env='#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"\nexport CUDA_HOME="@CUDA@"\nexport PATH="/opt/venvs/simct-b200/bin:$CUDA_HOME/bin:$PATH"\nexport LD_LIBRARY_PATH="/opt/simct-portable-libs:$CUDA_HOME/lib64:$(find /opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia -type d -name lib -printf \'%p:\' 2>/dev/null)${LD_LIBRARY_PATH:-}"\nexport KDFLOW_TRUST_REMOTE_CODE=0\nexport TOKENIZERS_PARALLELISM=false\nexec /opt/venvs/simct-b200/bin/python "$@"\n'
    (root/'python-b200.sh').write_text(run_env.replace('@CUDA@',str(cuda)))
    installer='#!/usr/bin/env bash\nset -euo pipefail\ncd -- "$(dirname -- "$0")"\nsha256sum -c SHA256SUMS\nwhile IFS= read -r target; do\n  if [ -e "$target" ] || [ -L "$target" ]; then\n    echo "Refusing to overwrite existing path: $target" >&2; exit 1\n  fi\ndone < targets.txt\n# Extract only to the reviewed, previously absent runtime paths.\ntar --keep-old-files -xzf runtime.tar.gz -C /\nmkdir source\ntar --keep-old-files -xf source.tar -C source\nbash python-b200.sh smoke.py\n'
    (root/'install.sh').write_text(installer)
    smoke="import json,torch,importlib.metadata as m\nassert torch.cuda.is_available(), 'CUDA driver not accessible'\nassert torch.cuda.device_count()==1, 'Expose only the selected GPU'\nassert torch.cuda.get_device_capability(0)==(10,0)\nx=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)\ny=x@x;torch.cuda.synchronize();assert torch.isfinite(y).all()\nprint(json.dumps({'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'sglang':m.version('sglang'),'transformers':m.version('transformers'),'status':'cuda_matmul_pass'}))\n"
    (root/'smoke.py').write_text(smoke)
    readme="""---
license: other
---
# SimCT B200 portable runtime
Private research runtime export, not a model or dataset. Source image contains
Ubuntu22.04 userspace; target Ubuntu24.04 x86_64, NVIDIA580.105.08/CUDA13.0.
This is a path-preserving runtime archive, not a relocatable virtualenv.
Install requires permission to create the paths in targets.txt (usually root).
Installer refuses existing paths and does not install/replace NVIDIA drivers.
Default visible GPU is0; GPU1 on the supplied host is occupied.
Download this pinned HF revision, then run `bash install.sh` in its directory.
It verifies SHA256 before extracting and runs a small B200 CUDA test.
Source is separate: `source/kdflow`, `source/experiments`.
For source imports set PYTHONPATH to `$PWD/source/experiments/modal/vendor:$PWD/source`.
Run Python with `bash python-b200.sh ...`; avoid relying on venv activation.
Model weights, datasets, credentials, caches and training outputs are excluded.
Host should provide bash/tar/gzip/sha256sum/find and gcc/g++ for JIT compilation.
No training auto-starts. The CUDA smoke uses only GPU0, unless explicitly overridden.
CPU packaging/import verification is not target GPU or full SGLang qualification.
The existing Phi Modal launcher is provenance, NOT a Qwen training launcher.
Licenses of bundled components remain in the runtime; consult them before reuse.
"""
    (root/'README.md').write_text(readme)
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.iterdir() if p.is_file() and p.name not in ('manifest.json','SHA256SUMS','receipt.json')}
    (root/'manifest.json').write_text(json.dumps(dict(image=IMAGE,source_commit=source_commit,files=files,targets=[str(x) for x in targets],target_gpu_test='pending user execution'),indent=2))
    (root/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(root.iterdir()) if p.is_file() and p.name not in ('SHA256SUMS','receipt.json')))
    subprocess.run(['bash','-n',str(root/'install.sh')],check=True)
    subprocess.run(['bash','-n',str(root/'python-b200.sh')],check=True)
    subprocess.run(['bash',str(root/'python-b200.sh'),'-c','import torch,transformers,sglang;print(torch.__version__,transformers.__version__,sglang.__version__)'],check=True,timeout=120)
    volume.commit();print('ARCHIVE_READY',json.dumps(files),flush=True)
    upload="""import os,json
from pathlib import Path
from huggingface_hub import HfApi
root=Path('/export/release-v1');api=HfApi(token=os.environ['HF_TOKEN'])
repo='codemaivanngu/simct-b200-portable-cu130'
assert api.whoami()['name']=='codemaivanngu'
if api.repo_exists(repo,repo_type='dataset'):raise RuntimeError('Existing repo requires inspection')
api.create_repo(repo,repo_type='dataset',private=True)
c=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=str(root),commit_message='Export pinned B200 runtime and source; target GPU check pending')
info=api.repo_info(repo,repo_type='dataset',revision=c.oid,files_metadata=True)
local={p.name:p for p in root.iterdir() if p.is_file()}
for f in info.siblings:
 if f.rfilename in local:
  assert f.size==local[f.rfilename].stat().st_size
  if f.lfs:
   import hashlib
   h=hashlib.sha256()
   with local[f.rfilename].open('rb') as stream:
    for b in iter(lambda:stream.read(8388608),b''):h.update(b)
   assert f.lfs.sha256==h.hexdigest()
receipt=dict(repo=repo,revision=c.oid,private=info.private,status='uploaded_size_and_lfs_hash_verified')
(root/'receipt.json').write_text(json.dumps(receipt));print('HF_RECEIPT',json.dumps(receipt),flush=True)
"""
    subprocess.run([python,'-c',upload],env={**os.environ,'HF_HUB_OFFLINE':'0'},check=True,timeout=600)
    volume.commit()

@app.function(image=image,cpu=4,memory=8192,timeout=900,retries=0,volumes={'/export':volume})
def upload_existing():
    code="import os,json\nfrom pathlib import Path\nfrom huggingface_hub import HfApi\nroot=Path('/export/release-v1');api=HfApi(token=os.environ['HF_TOKEN'])\nrepo='codemaivanngu/simct-b200-portable-cu130'\nassert api.whoami()['name']=='codemaivanngu'\nif api.repo_exists(repo,repo_type='dataset'):raise RuntimeError('Existing repo requires inspection')\napi.create_repo(repo,repo_type='dataset',private=True)\nc=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=str(root),commit_message='Export pinned B200 runtime and source; target GPU check pending')\ninfo=api.repo_info(repo,repo_type='dataset',revision=c.oid,files_metadata=True)\nlocal={p.name:p for p in root.iterdir() if p.is_file()}\nfor f in info.siblings:\n if f.rfilename in local:\n  assert f.size==local[f.rfilename].stat().st_size\n  if f.lfs:\n   import hashlib\n   h=hashlib.sha256()\n   with local[f.rfilename].open('rb') as stream:\n    for b in iter(lambda:stream.read(8388608),b''):h.update(b)\n   assert f.lfs.sha256==h.hexdigest()\nreceipt=dict(repo=repo,revision=c.oid,private=info.private,status='uploaded_size_and_lfs_hash_verified')\n(root/'receipt.json').write_text(json.dumps(receipt));print('HF_RECEIPT',json.dumps(receipt),flush=True)\n"
    subprocess.run(['/opt/venvs/simct-b200/bin/python','-c',code],env={**os.environ,'HF_HUB_OFFLINE':'0'},check=True,timeout=800)
    volume.commit()

@app.local_entrypoint()
def main(stage:str='inspect'):
    if stage=='inspect':inspect.remote();return
    root=Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise RuntimeError('Commit first')
    values={}
    for line in Path('/home/tung/Collaborative-MORL/.secrets/talapas_secrets.env').read_text().splitlines():
        line=line.removeprefix('export ')
        if line.startswith('HF_TOKEN='):values['HF_TOKEN']=shlex.split(line.split('=',1)[1])[0]
    secret=modal.Secret.from_dict(values)
    call=upload_existing.with_options(secrets=[secret]).spawn() if stage=='upload' else package.with_options(secrets=[secret]).spawn('41eb5b3')
    print('SUBMITTED',call.object_id,flush=True)
