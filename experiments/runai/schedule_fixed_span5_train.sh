#!/usr/bin/env bash
# Schedule the fixed span-5 ladder and the seed-44 re-runs, one stream per GPU.
#
# usage: schedule_fixed_span5_train.sh <gpuA> <gpuB> [min_free_mib]   (default 0 3 16384)
#   stream A (gpuA): FIX-fixed5-s42 -> FIX-fixed5-s43 -> FIX-fixed5-s44
#   stream B (gpuB): FIX-fixed3-s44 -> FIX-fixed4-s44
#
# Both seed-44 runs restart from scratch. They live in a fresh case, so MP_RESUME resolves to
# 0 with no directory surgery, and the interrupted partial runs in the old case stay untouched
# as evidence. The case is initialised from this checkout, which is also what runs the
# training, so the campaign commit gate cannot drift.
#
# Placement is per stream: each stream waits only for its own card, so a GPU occupied by
# another tenant does not hold back the other stream.
#
# Headroom: the six finished ladder runs peaked at 136.75..144.03 GiB reserved on a
# 179.06 GiB card, so a card already holding about 16 GiB is still usable, which is the
# default threshold. Training and evaluation cannot share a card (~144 + ~150 > 179).
#
# MP5_DRY=1 prints the resolved paths and exits without touching anything.
set -o pipefail

MP5_DRY="$MP5_DRY"
BASE="$MP5_BASE"
[ -n "$BASE" ] || BASE=/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM
OLD_CASE="$MP5_OLD_CASE"
[ -n "$OLD_CASE" ] || OLD_CASE=$BASE/fixedspan-ladder-20260929
NEW_CASE="$MP5_NEW_CASE"
[ -n "$NEW_CASE" ] || NEW_CASE=$BASE/fixedspan5-ladder-20260930
GPU_A="$1"
[ -n "$GPU_A" ] || GPU_A=0
GPU_B="$2"
[ -n "$GPU_B" ] || GPU_B=3
MIN_FREE_MIB="$3"
[ -n "$MIN_FREE_MIB" ] || MIN_FREE_MIB=16384
[ "$GPU_A" = "$GPU_B" ] && { echo "hai luong phai khac GPU"; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../.." && pwd)
PY=/usr/bin/python3.12
LOGS=$NEW_CASE/logs
WAIT_GPU_ROUNDS=72

echo "=== 0. source gate ==="
git -C "$SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "khong thay checkout git tai $SRC"; exit 2; }
echo "  SRC=$SRC"
echo "  HEAD=$(git -C "$SRC" rev-parse HEAD)"
echo "  dirty_files=$(git -C "$SRC" status --porcelain --untracked-files=no | wc -l)"
if [ "$MP5_DRY" = "1" ]; then
  echo "  DRY gpuA=$GPU_A gpuB=$GPU_B min_free_mib=$MIN_FREE_MIB"
  echo "  DRY old_case=$OLD_CASE"
  echo "  DRY new_case=$NEW_CASE"
  echo "  DRY stream A = FIX-fixed5-s42 FIX-fixed5-s43 FIX-fixed5-s44"
  echo "  DRY stream B = FIX-fixed3-s44 FIX-fixed4-s44"
  exit 0
fi

mkdir -p "$LOGS" "$NEW_CASE/captures"

echo
echo "=== 1. doi eval sinh xong (khong con worker/shard) ==="
while pgrep -f 'eval_queue.py worker' >/dev/null || pgrep -f 'shard4.sh' >/dev/null; do
  W=$(pgrep -c -f 'eval_queue.py worker' || true)
  S=$(pgrep -c -f 'shard4.sh' || true)
  echo "  $(date -Is) eval dang chay: worker=$W shard=$S"
  sleep 60
done
echo "  $(date -Is) EVAL_GEN_DONE"

echo
echo "=== 2. tao case moi (neu chua co) ==="
if [ ! -f "$NEW_CASE/campaign.json" ]; then
  STUDENT=$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["student"])' "$OLD_CASE/campaign.json")
  TEACHER=$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["teacher"])' "$OLD_CASE/campaign.json")
  DATASET=$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["dataset"])' "$OLD_CASE/campaign.json")
  PYTHONPATH="$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai" \
    $PY "$SRC/experiments/runai/queue_fixed_span_ladder.py" init \
      --case "$NEW_CASE" --template-case "$OLD_CASE" \
      --student "$STUDENT" --teacher "$TEACHER" --dataset "$DATASET" || exit 4
else
  echo "  case da co campaign.json, kiem commit ben duoi"
fi
$PY -c 'import json,sys;d=json.load(open(sys.argv[1]));print("  case commit",d["commit"][:7],"| runs",len(d["runs"]))' "$NEW_CASE/campaign.json"

echo
echo "=== 3. ghi 2 script luong ==="
cat > "$LOGS/stream_span5.sh" <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
SRC="$1"; CASE="$2"; SLOT="$3"
export PYTHONPATH="$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai"
export MP_LADDER_SLOTS="1,2,$SLOT"
export MP_PARITY_CAPTURE_DIR="$CASE/captures"
mkdir -p "$MP_PARITY_CAPTURE_DIR"
for id in FIX-fixed5-s42 FIX-fixed5-s43 FIX-fixed5-s44; do
  echo "=== START $id $(date -Is)"
  /usr/bin/python3.12 "$SRC/experiments/runai/queue_fixed_span_ladder.py" train-one --case "$CASE" --id "$id"
  echo "=== END $id rc=$? $(date -Is)"
done
echo STREAM_SPAN5_DONE $(date -Is)
EOS
cat > "$LOGS/stream_s44.sh" <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
SRC="$1"; CASE="$2"; SLOT="$3"
export PYTHONPATH="$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai"
export MP_PARITY_CAPTURE_DIR="$CASE/captures"
mkdir -p "$MP_PARITY_CAPTURE_DIR"
echo "=== START FIX-fixed3-s44 $(date -Is)"
MP_LADDER_SLOTS="$SLOT,2,1" /usr/bin/python3.12 "$SRC/experiments/runai/queue_fixed_span_ladder.py" train-one --case "$CASE" --id FIX-fixed3-s44
echo "=== END FIX-fixed3-s44 rc=$? $(date -Is)"
echo "=== START FIX-fixed4-s44 $(date -Is)"
MP_LADDER_SLOTS="2,$SLOT,1" /usr/bin/python3.12 "$SRC/experiments/runai/queue_fixed_span_ladder.py" train-one --case "$CASE" --id FIX-fixed4-s44
echo "=== END FIX-fixed4-s44 rc=$? $(date -Is)"
echo STREAM_S44_DONE $(date -Is)
EOS
chmod +x "$LOGS/stream_span5.sh" "$LOGS/stream_s44.sh"
bash -n "$LOGS/stream_span5.sh" && bash -n "$LOGS/stream_s44.sh" && echo "  STREAM_SCRIPTS_OK"

echo
echo "=== 4. moi luong cho GPU cua no roi tu bat ==="
starter() {
  local gpu="$1" script="$2" log="$3" tag="GPU$1" i=0 used
  while [ "$i" -lt "$WAIT_GPU_ROUNDS" ]; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    echo "  $(date -Is) $tag used=$used MiB (can < $MIN_FREE_MIB)"
    if [ "$used" -lt "$MIN_FREE_MIB" ]; then
      echo "  $(date -Is) $tag READY"
      setsid nohup bash "$script" "$SRC" "$NEW_CASE" "$gpu" >> "$log" 2>&1 < /dev/null &
      return 0
    fi
    i=$((i + 1))
    sleep 300
  done
  echo "  $(date -Is) $tag CHUA_DU_CHO sau $WAIT_GPU_ROUNDS lan thu"
  return 1
}
starter "$GPU_A" "$LOGS/stream_span5.sh" "$LOGS/stream_span5.out" &
PID_A=$!
starter "$GPU_B" "$LOGS/stream_s44.sh" "$LOGS/stream_s44.out" &
PID_B=$!
wait "$PID_A"
RC_A=$?
wait "$PID_B"
RC_B=$?

echo
echo "=== 5. ket qua xep lich ==="
echo "  stream span5 (gpu$GPU_A) rc=$RC_A"
echo "  stream s44   (gpu$GPU_B) rc=$RC_B"
if [ "$RC_B" -ne 0 ]; then
  echo "  GPU$GPU_B dang bi tien trinh khac chiem: chay lai voi GPU khac, vi du"
  echo "    bash $0 $GPU_A 1"
fi
sleep 45
echo "--- span5 ---"; tail -4 "$LOGS/stream_span5.out" 2>/dev/null
echo "--- s44 ---"; tail -4 "$LOGS/stream_s44.out" 2>/dev/null
echo "--- GPU ---"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "SCHEDULE_DONE $(date -Is)"
