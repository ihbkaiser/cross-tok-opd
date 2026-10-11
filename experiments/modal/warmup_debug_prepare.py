"""Remote-only selective checkpoint import with launch-hash verification."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import zipfile

from warmup_zip_access import RemoteZipReader, extract_member, REVISION, ARCHIVE_SHA256

ROOT=Path('/warmup')
PREFIXES={'phi':'phi4-gemma-sft-20261002-131247/checkpoint/',
          'qwen':'qwen-gemma-sft-paper-20260908-045828/checkpoint/'}
EXPECTED={'phi':'fb54dee22121b73f21260699b7b2ea64779a04e54dc0cd771168c140bb69414e',
          'qwen':'26a856f4a6e913f21e961cdcb5c7894f5966733e44f8b4624580de8f7ad439c1'}

def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'ready.json').exists():
        ready=json.loads((ROOT/'ready.json').read_text())
        assert ready['revision']==REVISION
        print('WARMUP_ASSETS_READY_JSON='+json.dumps(ready),flush=True);return
    reader=RemoteZipReader()
    with zipfile.ZipFile(reader) as z:
        members=[i for i in z.infolist() if not i.is_dir() and any(i.filename.startswith(p) and '/' not in i.filename[len(p):] for p in PREFIXES.values())]
    def fetch(info):
        result=extract_member(info,ROOT)
        print('ASSET_IMPORTED_JSON='+json.dumps(result),flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        receipts=list(pool.map(fetch,members))
    for arm,prefix in PREFIXES.items():
        receipt=next(r for r in receipts if r['member']==prefix+'model.safetensors')
        assert receipt['sha256']==EXPECTED[arm],(arm,receipt['sha256'])
    ready={'revision':REVISION,'archive_lfs_sha256':ARCHIVE_SHA256,'archive_full_hash_computed':False,
      'verification':'Selected members pass ZIP CRC and model weights match production launch SHA256; complete archive not downloaded.',
      'models':{arm:str(ROOT/prefix) for arm,prefix in PREFIXES.items()},'receipts':receipts}
    (ROOT/'ready.json').write_text(json.dumps(ready,indent=2))
    print('WARMUP_ASSETS_READY_JSON='+json.dumps(ready),flush=True)

if __name__=='__main__':main()
