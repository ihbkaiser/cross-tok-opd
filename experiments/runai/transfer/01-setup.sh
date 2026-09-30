#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PACKAGE"
/usr/bin/python3.12 - <<'PY'
import json,hashlib
from pathlib import Path
m=json.loads(Path('source-manifest.json').read_text())
assert hashlib.sha256(Path(m['bundle_file']).read_bytes()).hexdigest()==m['bundle_sha256'], 'bundle checksum mismatch'
print('BUNDLE_SHA256_PASS')
PY
git bundle verify "$PACKAGE/source.bundle"
WORK="$(mktemp -d /workspace/storage-shared/nlp/tungks/mp-opd-abc-XXXXXXXX)"
git clone --branch vdt/ops/b200-portable "$PACKAGE/source.bundle" "$WORK/source"
EXPECTED="$(/usr/bin/python3.12 -c 'import json; print(json.load(open("source-manifest.json"))["source_commit"])')"
[[ "$(git -C "$WORK/source" rev-parse HEAD)" == "$EXPECTED" ]]
printf 'export ABC_WORK=%q\n' "$WORK" > "$PACKAGE/workspace.env"
echo "CODE=$WORK/source"
echo "NEXT: bash $PACKAGE/02-prepare-mechanics.sh"
