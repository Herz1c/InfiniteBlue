#!/bin/bash
# Unattended M2 gate (PROGRESS E6, pre-registered 2026-10-17 18:58). Each step checks the previous
# step's output and stops loudly if it is missing. Safe to stop and restart: finished steps are skipped,
# and match.py resumes from its output file.
set -u
ROOT=/mnt/c/Users/honza/Desktop/InfiniteChess
RUN=~/data/runs/m2a
PY=~/venv/ic/bin/python
APEIRON=~/engines/apeiron-v7.10.0/sprt
OUT=$ROOT/data/matches
need() { [ -s "$1" ] || { echo "MISSING $1, stopping"; exit 1; }; }

echo "[$(date +%H:%M)] waiting for $RUN/DONE"
until [ -f "$RUN/DONE" ]; do sleep 300; done

need "$RUN/log.jsonl"
SEL=$($PY "$ROOT/tools/select_ckpt.py" "$RUN") || { echo "selection failed, stopping"; exit 1; }
need "$SEL"; need "$RUN/selected.json"
echo "[$(date +%H:%M)] selected $SEL"; cat "$RUN/selected.json"

ENGINE="$PY $ROOT/py/engine.py --ckpt $SEL --mode policy"
cd "$ROOT/py"
echo "[$(date +%H:%M)] SPRT vs Apeiron L1 (openings 0-199)"
$PY match.py --a "$ENGINE" --a-name m2-policy --b "$APEIRON" --b-strength 1 \
  --openings "$ROOT/bench/openings_classical_v1.jsonl" --pairs 200 --ms 1000 --concurrency 4 \
  --sprt 0 70.4 --min-pairs 20 --out "$OUT/m2_gate_sprt.jsonl" | tail -n 3
need "$OUT/m2_gate_sprt.jsonl"
echo "[$(date +%H:%M)] confirmation (openings 200-399)"
$PY match.py --a "$ENGINE" --a-name m2-policy --b "$APEIRON" --b-strength 1 \
  --openings "$ROOT/bench/openings_classical_v1.jsonl" --opening-offset 200 --pairs 200 --ms 1000 \
  --concurrency 4 --sprt 0 70.4 --min-pairs 100000 --out "$OUT/m2_gate_confirm.jsonl" | tail -n 3
need "$OUT/m2_gate_confirm.jsonl"
echo "[$(date +%H:%M)] M2 chain finished"
