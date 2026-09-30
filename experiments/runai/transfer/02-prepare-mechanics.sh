#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PACKAGE/workspace.env"
EXISTING=/workspace/storage-shared/nlp/tungks/SimCT/data/qwen-author/data/selected.parquet
mkdir -p "$ABC_WORK/data"
if [[ -f "$EXISTING" ]]; then
  DATA="$EXISTING"
else
  DATA="$ABC_WORK/data/selected.parquet"
  curl --fail --location --retry 3 --proxy http://10.30.154.118:80 --noproxy "" \
    'https://huggingface.co/datasets/codemaivanngu/simct-author-code-10k-qwen25-7b-instruct/resolve/1d276028899f515328e2074e01c97e1b03a89b5b/data/selected.parquet' \
    -o "$DATA"
fi
printf '%s  %s\n' af71af9d4a53103dc0e15983a34664dc3d43d0b944d3497997f8c0128ac5ec04 "$DATA" | sha256sum -c -
cd "$ABC_WORK/source"
/usr/bin/python3.12 experiments/mp_opd/real_oracle.py prepare \
  --input "$DATA" --reference-key label --conflicting-references exclude --groups 2 --seed 42 \
  --reference-provenance 'HF 1d276028899f515328e2074e01c97e1b03a89b5b selected; potential SFT/train exposure; mechanics only' \
  --output "$ABC_WORK/data/seen-sft-mechanics-2groups.json"
echo 'PREPARED: mechanics only, not unseen-data scientific evidence; no GPU started.'
echo "NEXT after choosing a free GPU: bash $PACKAGE/03-run-mechanics.sh GPU_SLOT"
