"""Selective remote import of flat TRUST/ALIGN checkpoint ZIP members.

CRC checks detect transfer corruption; recorded member SHA hashes are provenance,
not independent expected hashes. The outer LFS SHA is not recomputed.
"""
import json
from pathlib import Path
import re
import zipfile

import warmup_zip_access as access
from checkpoint_replay_assets import download_stored_member

REVISION = '010c9a418f167c4bae6fb7ee07fa838fc7583bb8'
ARCHIVE = 'degen-round1-all-20261011.zip'
ARCHIVE_BYTES = 51541871208
LFS_SHA256 = '227e127a3e060cdaea864787a18ae66f3a04d9e0c143a2645d1e6480429f2d76'
REQUIRED = {'model.safetensors','config.json','generation_config.json',
            'tokenizer.json','tokenizer_config.json','chat_template.jinja'}


def checkpoint_groups(infos):
    groups = {}
    for info in infos:
        if info.is_dir():
            continue
        match = re.fullmatch(r'(simct-runs/qwen-gemma-mp_opd-(trust_r|trust_b|align)-[^/]+/checkpoint/step(\d+))/(.+)',info.filename)
        if not match or match[4] not in REQUIRED:
            raise ValueError('Unexpected flat checkpoint member: '+info.filename)
        arm = match[2].replace('_','-')+'-step'+match[3]
        row = groups.setdefault(arm, {'source_directory':match[1], 'infos':[]})
        if row['source_directory'] != match[1]:
            raise ValueError('Ambiguous run provenance for '+arm)
        row['infos'].append(info)
    for arm,group in groups.items():
        names = [Path(i.filename).name for i in group['infos']]
        if len(names)!=len(REQUIRED) or set(names)!=REQUIRED:
            raise ValueError('Missing or duplicate checkpoint member: '+arm)
    return groups


def prepare(root=Path('/checkpoints')):
    access.ARCHIVE_BYTES = ARCHIVE_BYTES
    access.URL = f'https://huggingface.co/datasets/{access.REPO}/resolve/{REVISION}/{ARCHIVE}'
    target = root/ARCHIVE.removesuffix('.zip')
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(access.RemoteZipReader()) as outer:
        groups = checkpoint_groups(outer.infolist())
    models = {}
    for arm, group in groups.items():
        receipt_file = target/(arm+'-receipt.json')
        if receipt_file.exists():
            models[arm]=json.loads(receipt_file.read_text())
            print('REUSED_CHECKPOINT='+arm,flush=True)
            continue
        members=[]
        for info in group['infos']:
            print('IMPORTING_MEMBER='+json.dumps({'arm':arm,'name':Path(info.filename).name,'bytes':info.file_size}),flush=True)
            row=download_stored_member(info,target)
            members.append({'name':info.filename,'bytes':row['bytes'],'sha256':row['sha256'],
                            'zip_crc_pass':row['zip_crc_pass'],'imported':True})
        model_path=target/group['source_directory']
        config=json.loads((model_path/'config.json').read_text())
        if config['model_type']!='gemma2':raise RuntimeError('Unexpected architecture')
        result={'arm':arm,'model_path':str(model_path),'source_directory':group['source_directory'],
                'archive':ARCHIVE,'revision':REVISION,'outer_lfs_sha256':LFS_SHA256,
                'outer_full_hash_computed':False,'independent_member_hash_available':False,
                'members':members,'config':config}
        receipt_file.write_text(json.dumps(result,indent=2))
        models[arm]=result
        print('IMPORTED_CHECKPOINT='+json.dumps({'arm':arm,'model_path':str(model_path),'files':len(members)}),flush=True)
    (root/'ready.json').write_text(json.dumps({'models':models,'revision':REVISION},indent=2))
    print('ALL_CHECKPOINTS_READY='+json.dumps(list(models)),flush=True)


if __name__=='__main__':prepare()
