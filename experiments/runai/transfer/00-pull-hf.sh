#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/storage-shared/nlp/tungks
WORK="$(mktemp -d "$ROOT/mp-opd-abc-XXXXXXXX")"
PROXY="${MP_DOWNLOAD_PROXY:-http://10.30.154.118:80}"
fetch() { curl --fail --location --retry 3 --proxy "$PROXY" --noproxy "" "$1" -o "$2"; }
if [[ -z "${HF_REVISION:-}" ]]; then
  fetch 'https://huggingface.co/api/models/codemaivanngu/simct' "$WORK/hf-info.json"
  HF_REVISION="$(/usr/bin/python3.12 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha"])' "$WORK/hf-info.json")"
fi
[[ "$HF_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid HF revision'; exit 2; }
BASE="https://huggingface.co/codemaivanngu/simct/resolve/$HF_REVISION"
fetch "$BASE/source-manifest.json" "$WORK/source-manifest.json"
fetch "$BASE/simct-b200-portable.bundle" "$WORK/simct-b200-portable.bundle"
/usr/bin/python3.12 - "$WORK" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); m=json.loads((root/'source-manifest.json').read_text())
assert m['bundle_file']=='simct-b200-portable.bundle'
assert hashlib.sha256((root/m['bundle_file']).read_bytes()).hexdigest()==m['bundle_sha256'], 'bundle checksum mismatch'
assert m['source_branch']=='vdt/ops/b200-portable'
print('BUNDLE_SHA256_PASS')
PY
git clone --branch vdt/ops/b200-portable "$WORK/simct-b200-portable.bundle" "$WORK/source"
EXPECTED="$(/usr/bin/python3.12 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$WORK/source-manifest.json")"
[[ "$(git -C "$WORK/source" rev-parse HEAD)" == "$EXPECTED" ]]
for NAME in 02-prepare-mechanics.sh 03-run-mechanics.sh; do
  cp "$WORK/source/experiments/runai/transfer/$NAME" "$WORK/$NAME"
done
printf 'export ABC_WORK=%q\n' "$WORK" > "$WORK/workspace.env"
printf 'HF_REVISION=%s\nSOURCE_COMMIT=%s\nABC_WORK=%s\n' "$HF_REVISION" "$EXPECTED" "$WORK"
echo "NEXT: bash $WORK/02-prepare-mechanics.sh"
echo 'No GPU started; existing training checkout unchanged.'
