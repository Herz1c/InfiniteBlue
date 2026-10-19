#!/bin/bash
# Unattended M3 gate (PROGRESS E7, pre-registered 2026-10-17 18:58; run settings in amendment A3).
# Waits until data generation has finished (so Apeiron and our search do not compete with 16
# data_gen threads for the CPU), then: SPRT at 1 s/move on test openings 0-199, then a 200-pair
# confirmation at 5 s/move on openings 200-249 (A4), then the exploratory E7b match vs L6. Safe to stop and restart: match.py resumes from
# its output file.
set -u
ROOT=/mnt/c/Users/honza/Desktop/InfiniteChess
RUN=~/data/runs/m2a
PY=~/venv/ic/bin/python
APEIRON=~/engines/apeiron-v7.10.0/sprt
OUT=$ROOT/data/matches
need() { [ -s "$1" ] || { echo "MISSING $1, stopping"; exit 1; }; }
# A pid file only counts if that pid is still the process we started: pids are reused (e.g. after a
# WSL restart), so the process start time is saved next to the pid and compared.
started_at() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f20; }
alive() { local f=~/jobs/$1; [ -f "$f.pid" ] && [ -n "$(started_at "$(cat "$f.pid")")" ] && [ "$(started_at "$(cat "$f.pid")")" = "$(cat "$f.start" 2>/dev/null)" ]; }
# A stop request: the running match.py records nothing in progress and exits; the chain stops too.
failed() { echo "[$(date +%H:%M)] match.py failed (exit $1), stopping"; exit 1; }
stopped() { echo "[$(date +%H:%M)] stopped by request (resume: bash tools/jobs.sh start m3gate)"; exit 130; }
trap stopped INT TERM
# match.py exits with 130 on a stop request (bash itself ignores the SIGINT then, so check the code);
# any other non-zero exit stops the chain loudly.

echo "[$(date +%H:%M)] waiting for job gen to finish"
while alive gen; do sleep 120; done

need "$RUN/selected.json"
SEL=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$RUN/selected.json")
need "$SEL"
echo "[$(date +%H:%M)] net $SEL (the E6 selection)"

# Search parameters are engine.py's frozen defaults: c_puct 2.0, FPU 0.3, contempt 0, batch 64.
ENGINE="$PY $ROOT/py/engine.py --ckpt $SEL --mode search"
cd "$ROOT/py"
echo "[$(date +%H:%M)] SPRT vs Apeiron L4, 1 s/move (openings 0-199)"
$PY match.py --a "$ENGINE" --a-name m3-mcts --b "$APEIRON" --b-strength 4 \
  --openings "$ROOT/bench/openings_classical_v1.jsonl" --pairs 200 --ms 1000 --concurrency 4 \
  --sprt 0 34.9 --min-pairs 20 --out "$OUT/m3_gate_sprt.jsonl" | tail -n 3 | tee "$OUT/m3_gate_sprt.result"
rc=${PIPESTATUS[0]}; [ "$rc" = 130 ] && stopped; [ "$rc" = 0 ] || failed "$rc"
need "$OUT/m3_gate_sprt.jsonl"
# E6/E7 rule: H0 accepted by the SPRT means REJECTED, so the 5 s confirmation is not needed.
if grep -q "H0 accepted" "$OUT/m3_gate_sprt.result"; then echo "[$(date +%H:%M)] SPRT accepted H0: H5 REJECTED, no confirmation run"; exit 0; fi
echo "[$(date +%H:%M)] confirmation, 5 s/move (openings 200-249, A4)"
$PY match.py --a "$ENGINE" --a-name m3-mcts --b "$APEIRON" --b-strength 4 \
  --openings "$ROOT/bench/openings_classical_v1.jsonl" --opening-offset 200 --pairs 50 --ms 5000 \
  --concurrency 4 --sprt 0 34.9 --min-pairs 100000 --out "$OUT/m3_gate_confirm.jsonl" | tail -n 3
rc=${PIPESTATUS[0]}; [ "$rc" = 130 ] && stopped; [ "$rc" = 0 ] || failed "$rc"
need "$OUT/m3_gate_confirm.jsonl"
# The confirmation above is 50 pairs since amendment A4.
# E7b (amendment A4): exploratory, descriptive only; Apeiron level 6 on fresh openings 400-449.
echo "[$(date +%H:%M)] E7b exploratory: vs Apeiron L6, 5 s/move (openings 400-449)"
$PY match.py --a "$ENGINE" --a-name m3-mcts --b "$APEIRON" --b-strength 6   --openings "$ROOT/bench/openings_classical_v1.jsonl" --opening-offset 400 --pairs 50 --ms 5000   --concurrency 4 --out "$OUT/m3_explore_L6.jsonl" | tail -n 2
rc=${PIPESTATUS[0]}; [ "$rc" = 130 ] && stopped; [ "$rc" = 0 ] || failed "$rc"
need "$OUT/m3_explore_L6.jsonl"
echo "[$(date +%H:%M)] M3 chain finished"
