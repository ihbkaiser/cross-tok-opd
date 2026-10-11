"""Import eight pinned checkpoint ZIPs remotely, validating sidecar SHA and CRC."""
import hashlib
import binascii
import json
from pathlib import Path
import shutil
import struct
import zipfile
import requests

import warmup_zip_access as access

REVISION = '7a80c71eb4cf86a05ee63e5b64d5f3e960abcd03'
SPECS = [
    ('simct-round1-all-20261011.zip', 25770933408,
     '5c4379c4a955787b35f7b98f12d3650ce0be82c76074b58107ef036e808ac169'),
    ('temporal-native-rl2-s44-all-20261011.zip', 25770933516,
     '4f63922c3a61b3c7545e3b49bd2b50e334718122f657749119795d0d342dde0c'),
]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_target(root, name):
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('ZIP member escapes extraction root')
    return target


def download_stored_member(info,root):
    """Resume a stored nested ZIP after a dropped HTTP connection, then hash/CRC it."""
    if info.compress_type!=zipfile.ZIP_STORED:raise ValueError('Resume requires a stored outer member')
    root=Path(root).resolve();target=safe_target(root,info.filename)
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_name(target.name+'.range-partial')
    chosen=target if target.exists() else partial
    digest=hashlib.sha256();crc=0;written=0
    if chosen.exists():
        with chosen.open('rb') as source:
            for chunk in iter(lambda:source.read(4*1024*1024),b''):
                digest.update(chunk);crc=binascii.crc32(chunk,crc);written+=len(chunk)
        if written>info.file_size:raise RuntimeError('Existing partial exceeds member length')
    if target.exists() and written!=info.file_size:raise RuntimeError('Unexpected incomplete final ZIP')
    with access.checked_range(info.header_offset,info.header_offset+29) as response:
        values=struct.unpack('<4s5H3I2H',response.content)
    if values[0]!=b'PK\x03\x04':raise RuntimeError('Invalid local ZIP header')
    start=info.header_offset+30+values[-2]+values[-1]
    failures=0
    if not target.exists():
        with partial.open('ab') as output:
            while written<info.file_size:
                end=min(info.file_size,written+64*1024*1024)
                try:
                    with access.checked_range(start+written,start+end-1) as response:
                        for chunk in response.iter_content(4*1024*1024):
                            if written+len(chunk)>end:raise RuntimeError('HTTP body exceeds requested range')
                            output.write(chunk);digest.update(chunk);crc=binascii.crc32(chunk,crc);written+=len(chunk)
                    if written!=end:raise RuntimeError('Incomplete range without transport error')
                    failures=0
                except requests.RequestException:
                    failures+=1;output.flush()
                    if failures>=3:raise
                    print('RESUMING_RANGE='+json.dumps({'member':info.filename,'bytes':written}),flush=True)
    if written!=info.file_size or (crc & 0xffffffff)!=info.CRC:raise RuntimeError('Resumed ZIP size/CRC mismatch')
    if not target.exists():partial.replace(target)
    return {'member':info.filename,'bytes':written,'compressed_bytes':written,'sha256':digest.hexdigest(),
            'zip_crc_pass':True,'path':str(target)}


def prepare(root=Path('/checkpoints')):
    root.mkdir(parents=True, exist_ok=True)
    models = {}
    for archive, size, lfs_sha in SPECS:
        access.ARCHIVE_BYTES = size
        access.URL = f'https://huggingface.co/datasets/{access.REPO}/resolve/{REVISION}/{archive}'
        with zipfile.ZipFile(access.RemoteZipReader()) as outer:
            infos = outer.infolist()
        target_root = root / archive.removesuffix('.zip')
        target_root.mkdir(exist_ok=True)
        for info in infos:
            if info.filename.endswith('.sha256'):
                access.extract_member(info, target_root)
        for info in infos:
            if not info.filename.endswith('.zip'):
                continue
            arm = Path(info.filename).stem
            receipt_file = target_root / (arm + '-receipt.json')
            if receipt_file.exists():
                models[arm] = json.loads(receipt_file.read_text())
                print('REUSED_CHECKPOINT='+arm, flush=True)
                continue
            expected = (target_root / (info.filename + '.sha256')).read_text().split()[0]
            if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                raise ValueError('Invalid SHA256 sidecar')
            receipt = download_stored_member(info, target_root)
            if receipt['sha256'] != expected:
                raise RuntimeError('Inner ZIP SHA256 differs from supplied sidecar')
            inner_path = Path(receipt['path'])
            extracted = target_root / arm
            extracted.mkdir(exist_ok=True)
            members = []
            with zipfile.ZipFile(inner_path) as inner:
                for member in inner.infolist():
                    if member.is_dir():
                        continue
                    dest = safe_target(extracted, member.filename)
                    # Reading through ZipExtFile verifies every selected member CRC.
                    # Only weights/config/tokenizer and text provenance are imported.
                    allowed = dest.suffix in ('.safetensors', '.json', '.model', '.txt', '.md', '.jinja', '.log')
                    row = {'name': member.filename, 'bytes': member.file_size, 'imported': allowed}
                    if allowed:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with inner.open(member) as source, dest.open('wb') as output:
                            shutil.copyfileobj(source, output, 4*1024*1024)
                        row.update(sha256=sha(dest), zip_crc_pass=True)
                    members.append(row)
            configs = list(extracted.rglob('config.json'))
            candidates = [p.parent for p in configs if list(p.parent.glob('*.safetensors'))]
            if len(candidates) != 1:
                raise RuntimeError(f'Ambiguous checkpoint root: {arm}: {candidates}')
            model_path = candidates[0]
            config = json.loads((model_path/'config.json').read_text())
            if config['model_type'] != 'gemma2':
                raise RuntimeError('Unexpected model architecture')
            final = {'arm': arm, 'model_path': str(model_path), 'inner_zip': receipt,
                     'sidecar_sha256': expected, 'sidecar_hash_pass': True,
                     'archive': archive, 'revision': REVISION, 'outer_lfs_sha256': lfs_sha,
                     'outer_full_hash_computed': False, 'members': members, 'config': config}
            receipt_file.write_text(json.dumps(final, indent=2))
            models[arm] = final
            print('IMPORTED_CHECKPOINT='+json.dumps({'arm':arm,'model_path':str(model_path),
                'inner_sha256':expected,'files':len(members)}), flush=True)
    ready = {'models': models, 'revision': REVISION}
    (root/'ready.json').write_text(json.dumps(ready, indent=2))
    print('ALL_CHECKPOINTS_READY='+json.dumps(list(models)), flush=True)


if __name__ == '__main__':
    prepare()
