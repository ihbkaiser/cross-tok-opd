#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK="$(dirname "$SOURCE")"
DATA=/workspace/storage-shared/nlp/tungks/SimCT/data/qwen-author/data/selected.parquet
mkdir -p "$WORK/data"
if [[ ! -f "$DATA" ]]; then
  DATA="$WORK/data/selected.parquet"
  curl -fL --retry 3 --proxy http://10.30.154.118:80 --noproxy ""     'https://huggingface.co/datasets/codemaivanngu/simct-author-code-10k-qwen25-7b-instruct/resolve/1d276028899f515328e2074e01c97e1b03a89b5b/data/selected.parquet' -o "$DATA"
fi
printf '%s  %s\n' af71af9d4a53103dc0e15983a34664dc3d43d0b944d3497997f8c0128ac5ec04 "$DATA" | sha256sum -c -
/usr/bin/python3.12 "$SOURCE/experiments/mp_opd/real_oracle.py" prepare  --input "$DATA" --reference-key label --conflicting-references exclude  --groups 4 --seed 142 --select-references 8 --eval-references 4  --reference-provenance 'HF 1d276028899f515328e2074e01c97e1b03a89b5b selected; seen SFT mechanics, not unseen-data evidence'  --output "$WORK/data/multiref-mechanics.json"
echo "PREPARED=$WORK/data/multiref-mechanics.json"
echo 'No GPU started. This is seen-SFT mechanics data, not a scientific generalization gate.'
