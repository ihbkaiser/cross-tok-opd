"""Transport-only TUNA fallback for two incomplete PyPI CUDA wheels.

The canonical project and lock stay byte-identical. A private installation view
changes only wheel URLs, retaining registry identities, versions and hashes so
uv can reuse all other cached artifacts. uv verifies complete wheel hashes.
"""
import argparse
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import tomllib
from urllib.parse import unquote, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
import uuid

HERE = Path(__file__).resolve().parent
MIRROR = 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/'
PACKAGES = ('nvidia-cublas-cu12', 'nvidia-cudnn-cu12')


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)


def mirror_url(package, wheel, body):
    parser = Links()
    parser.feed(body)
    expected_name = unquote(urlparse(wheel['url']).path.rsplit('/', 1)[-1])
    expected_hash = wheel['hash'].replace('sha256:', 'sha256=')
    assert wheel['hash'].startswith('sha256:') and len(wheel['hash']) == 71
    matches = set()
    for href in parser.links:
        link = urljoin(MIRROR + package + '/', href)
        parsed = urlparse(link)
        if unquote(parsed.path.rsplit('/', 1)[-1]) != expected_name:
            continue
        assert parsed.scheme == 'https' and parsed.hostname == 'mirrors.tuna.tsinghua.edu.cn'
        assert parsed.username is None and parsed.password is None
        assert not parsed.query and parsed.port is None
        clean, fragment = urldefrag(link)
        assert fragment == expected_hash, 'Mirror advertised hash differs from canonical lock'
        matches.add(clean)
    assert len(matches) == 1, 'Exact mirror wheel is absent or ambiguous'
    return matches.pop()


def fetch_index(package):
    request = Request(MIRROR + package + '/', headers={
        'User-Agent': 'vdt-locked-mirror/1.0', 'Accept-Encoding': 'identity'})
    with urlopen(request, timeout=30) as response:
        assert response.status == 200
        final = urlparse(response.url)
        assert final.scheme == 'https' and final.hostname == 'mirrors.tuna.tsinghua.edu.cn'
        body = response.read(2 * 1024 * 1024 + 1)
    assert len(body) <= 2 * 1024 * 1024, 'Unexpectedly large mirror index'
    return body.decode('utf-8')


def transport_lock(original, fetch=fetch_index):
    parsed = tomllib.loads(original)
    packages = {p['name']: p for p in parsed['package']}
    result, mappings = original, []
    for name in PACKAGES:
        package = packages[name]
        assert package['source'] == {'registry': 'https://pypi.org/simple'}
        assert len(package['wheels']) == 1, 'Review platform-specific wheel selection first'
        wheel = package['wheels'][0]
        assert urlparse(wheel['url']).hostname == 'files.pythonhosted.org'
        replacement = mirror_url(name, wheel, fetch(name))
        assert result.count(wheel['url']) == 1 and replacement not in result
        result = result.replace(wheel['url'], replacement, 1)
        mappings.append(dict(package=name, version=package['version'],
                             original_url=wheel['url'], mirror_url=replacement,
                             sha256=wheel['hash'], size=wheel['size']))
    normalized = result
    for mapping in mappings:
        normalized = normalized.replace(mapping['mirror_url'], mapping['original_url'], 1)
    assert normalized == original, 'Change outside the two transport URLs'
    mirrored = tomllib.loads(result)
    by_name = {p['name']: p for p in mirrored['package']}
    for mapping in mappings:
        by_name[mapping['package']]['wheels'][0]['url'] = mapping['original_url']
    assert mirrored == parsed, 'Resolved dependency graph changed'
    return result, mappings


def run(root, project=HERE/'xtoken'):
    root = root.resolve(strict=True)
    project = project.resolve(strict=True)
    assert root not in (Path('/'), Path.home())
    canonical = {name: (project/name).read_bytes()
                 for name in ('pyproject.toml', 'uv.lock', '.python-version')}
    changed, mappings = transport_lock(canonical['uv.lock'].decode())
    parent = root/'mirror-installs'
    parent.mkdir(exist_ok=True, mode=0o700)
    out = parent/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    out.mkdir(mode=0o700)
    view = out/'project'
    view.mkdir(mode=0o700)
    for name, data in canonical.items():
        (view/name).write_bytes(changed.encode() if name == 'uv.lock' else data)
    report = dict(canonical_project=str(project), installation_view=str(view),
                  canonical_sha256={n: hashlib.sha256(b).hexdigest() for n, b in canonical.items()},
                  transport_lock_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                  mappings=mappings, normalized_lock_identical=True,
                  dependency_graph_identical=True, automatic_retry=False,
                  complete_wheel_hash_verification='enforced_by_uv')
    (out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(transport_manifest=str(out/'manifest.json'),
                          mirror_packages=list(PACKAGES), lock_identity='unchanged_except_transport')), flush=True)
    assert all((project/n).read_bytes() == b for n, b in canonical.items())
    # --frozen consumes the audited view without re-resolution or rewriting it.
    # UV_PROJECT_ENVIRONMENT points to the existing dedicated X-Token venv.
    assert Path(os.environ['UV_PROJECT_ENVIRONMENT']).resolve() == root/'xtoken/venv'
    command = ['uv', 'sync', '--frozen', '--project', str(view)]
    rc = subprocess.run(command, check=False).returncode
    assert all((project/n).read_bytes() == b for n, b in canonical.items())
    assert (view/'uv.lock').read_bytes() == changed.encode()
    report.update(exit_code=rc, canonical_unchanged_after_sync=True)
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    return rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    assert args.root.is_absolute()
    os.umask(0o077)
    raise SystemExit(run(args.root))
