#!/bin/bash
# M4 ladder (PROGRESS E12, tracking only): every training snapshot ckpt_<step>.pt plays Apeiron at
# full strength (L8) on DEV openings 0-19, 20 pairs at 1 s/move (one game at a time, RAM cap), and its score is appended to
# ladder.jsonl in the run directory. Never touches the test suite. Run as job `ladder`; resumable
# (finished snapshots are skipped, a half-played match resumes from its games file).
set -u
ROOT=/mnt/c/Users/honza/Desktop/InfiniteChess
RUN=~/data/runs/${RUN:-m4}
PY=~/venv/ic/bin/python
APEIRON=~/engines/apeiron-v7.10.0/sprt
OUT=$ROOT/data/matches/m4_ladder
mkdir -p "$OUT"
stopped() { echo "[$(date +%H:%M)] stopped by request"; exit 130; }
trap stopped INT TERM
cd "$ROOT/py"
while true; do
  for ck in $(ls "$RUN"/ckpt_*.pt 2>/dev/null | sort); do
    step=$(basename "$ck" .pt | sed 's/ckpt_0*//')
    grep -q "\"step\": $step," "$RUN/ladder.jsonl" 2>/dev/null && continue
    echo "[$(date +%H:%M)] ladder: snapshot $step vs Apeiron L8 (dev 0-19, 1 s)"
    $PY match.py --a "$PY $ROOT/py/engine.py --ckpt $ck --mode search" --a-name m4-$step --b "$APEIRON" \
      --openings "$ROOT/bench/openings_dev_v1.jsonl" --pairs 20 --ms 1000 --concurrency 1 \
      --out "$OUT/ladder_$step.jsonl" | tail -n 1 > "$OUT/ladder_$step.result"
    rc=${PIPESTATUS[0]}; [ "$rc" = 130 ] && stopped
    [ "$rc" = 0 ] || { echo "match.py failed (exit $rc) for snapshot $step, stopping"; exit 1; }
    score=$(grep -o 'score [0-9.]*' "$OUT/ladder_$step.result" | cut -d' ' -f2)
    echo "{\"step\": $step, \"score\": $score, \"result\": \"$(cat "$OUT/ladder_$step.result")\"}" >> "$RUN/ladder.jsonl"
    echo "[$(date +%H:%M)] LADDER step $step: $(cat "$OUT/ladder_$step.result")"
  done
  sleep 300 & wait $!
done
