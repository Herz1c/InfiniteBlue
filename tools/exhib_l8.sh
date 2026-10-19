#!/bin/bash
# Exploratory match requested by the user (2026-10-19): at 20:00 the latest self-play net plays
# Apeiron at full strength (L8), 5 s/move, 10 pairs on DEV openings 20-29 (not the test suite);
# then the pre-RL net (m2a) plays the same match for comparison. Self-play, training and the ladder
# are paused during the matches so our engine has the whole GPU, and resumed afterwards.
# Run as job `exhib`; resumable (match.py keeps finished games; a frozen copy of the net is used).
set -u
ROOT=/mnt/c/Users/honza/Desktop/InfiniteChess
PY=~/venv/ic/bin/python
APEIRON=~/engines/apeiron-v7.10.0/sprt
J="bash $ROOT/tools/jobs.sh"
OUT=$ROOT/data/matches
NET=~/data/runs/m4/exhib_2000.pt
stopped() { echo "[$(date +%H:%M)] stopped by request (resume: bash tools/jobs.sh start exhib)"; exit 130; }
trap stopped INT TERM

until [ "$(date +%H%M)" -ge 2000 ]; do sleep 60 & wait $!; done
if [ ! -s "$NET" ]; then
  cp ~/data/runs/m4/current.pt "$NET.tmp" && mv "$NET.tmp" "$NET"
fi
step=$($PY -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu").get("step"))' "$NET")
echo "[$(date +%H:%M)] frozen RL net at trainer step $step; pausing self-play, training and ladder"
for j in ladder sp rl; do $J stop $j; done

cd "$ROOT/py"
play() {  # name, checkpoint, output file
  echo "[$(date +%H:%M)] $1 vs Apeiron L8 (full strength), 5 s/move, dev openings 20-29"
  $PY match.py --a "$PY $ROOT/py/engine.py --ckpt $2 --mode search" --a-name "$1" --b "$APEIRON" \
    --openings "$ROOT/bench/openings_dev_v1.jsonl" --opening-offset 20 --pairs 10 --ms 5000 --concurrency 4 \
    --out "$3" | tail -n 1
  rc=${PIPESTATUS[0]}; [ "$rc" = 130 ] && stopped
  [ "$rc" = 0 ] || { echo "match.py failed (exit $rc), stopping"; exit 1; }
}
play "rl-$step" "$NET" "$OUT/exhib_L8_rl.jsonl"
play "m2a-base" ~/data/runs/m2a/ckpt_0200000.pt "$OUT/exhib_L8_base.jsonl"

echo "[$(date +%H:%M)] matches done; resuming training, self-play and ladder"
for j in rl sp ladder; do $J start $j; done
echo "[$(date +%H:%M)] exhib finished"
