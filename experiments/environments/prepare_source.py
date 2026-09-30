"""Fetch/verify the exact NeMo source archive recorded in the native uv lock."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import tomllib
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_tree(archive, target):
    records = []
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) == 1 or member.isdir():
                continue
            relative = Path(*parts[1:])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe archive path')
            path = target/relative
            if member.issym():
                if not path.is_symlink() or str(path.readlink()) != member.linkname:
                    raise ValueError(f'Source symlink differs: {relative}')
                records.append((str(relative), 'symlink:'+member.linkname))
            elif member.isfile():
                with tar.extractfile(member) as stream:
                    expected = hashlib.file_digest(stream, 'sha256').hexdigest()
                if path.is_symlink() or not path.is_file() or digest(path) != expected:
                    raise ValueError(f'Source file differs: {relative}')
                records.append((str(relative), expected))
            else:
                raise ValueError('unsupported archive member')
    return dict(files_verified=len(records), tree_sha256=hashlib.sha256(
        json.dumps(sorted(records), separators=(',', ':')).encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.root.is_absolute() or root in (Path('/'), Path.home()):
        parser.error('dedicated absolute runtime root required')
    root.mkdir(parents=True, exist_ok=True)
    lock = tomllib.loads((HERE/'xtoken/uv.lock').read_text())
    package = next(p for p in lock['package'] if p['name'] == 'nemo-rl')
    url, expected = package['source']['url'], package['sdist']['hash'].removeprefix('sha256:')
    target = root/'NeMo-RL'
    with tempfile.TemporaryDirectory(prefix='source-verify-', dir=root) as temporary:
        temp = Path(temporary)
        archive = temp/'source.tar.gz'
        with urlopen(url, timeout=60) as response, archive.open('wb') as stream:
            while chunk := response.read(1024*1024):
                stream.write(chunk)
        if digest(archive) != expected:
            raise ValueError('source archive checksum mismatch')
        if not target.exists():
            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall(temp/'unpacked', filter='data')
            children = list((temp/'unpacked').iterdir())
            if len(children) != 1 or not children[0].is_dir():
                raise ValueError('unexpected source archive layout')
            children[0].rename(target)
        proof = verify_tree(archive, target)
    proof.update(archive_url=url, archive_sha256=expected, source_modified=False,
                 upstream_lock_sha256=digest(target/'uv.lock'))
    (root/'NeMo-RL.provenance.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
