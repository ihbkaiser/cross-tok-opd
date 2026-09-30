"""Submit the historical LCBfix CPU job to the owner's existing manager."""
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM')


def main():
    if socket.gethostname()!='tungks-0-1':
        raise ValueError('Owner host only')
    cfg=json.loads((BASE/'owner-tungks-0-1/campaign.json').read_text())
    sys.path.insert(0,cfg['manager'])
    from job_manager.store import connect,rows,submit
    from job_manager.__main__ import snapshot
    db=connect(Path(cfg['state']))
    try:
        state=snapshot(db)
        if not state['manager']['running'] or state['paused'] or state['quarantined']:
            raise ValueError('Manager not ready')
        if any(j['id']=='lcbfix-v1' for j in rows(db)):
            print(json.dumps(state,indent=2));return
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
            raise ValueError('Dirty source')
        original=json.loads(Path('/workspace/storage-shared/nlp/tungks/campaign20-A7hEqD4H/work/config.json').read_text())
        spec=dict(version=1,id='lcbfix-v1',project='simct-lcbfix',cwd=str(ROOT),gpus=0,cpu_slots=4,
                  timeout_seconds=86400,dependencies=[],env={},
                  argv=[sys.executable,str(ROOT/'scripts/evaluation/lcbfix.py'),
                        '--plan',original['eval_template'],'--out',str(BASE/'lcbfix-v1'),
                        '--workers','4','--internal-code-execution'],
                  metadata=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                                scope='historical three-seed LCB extractor correction; CPU only'))
        submit(db,[spec])
        print(json.dumps(snapshot(db),indent=2))
    finally:
        db.close()


if __name__=='__main__':
    main()
