#!/bin/bash
# Prints one line per noteworthy event of the long jobs (for a log monitor): a job that stops
# running, training starting, each validation eval, every 10k training steps, and each line of
# the gate chains, and a running score every 20 finished gate games. Silent otherwise. Run inside WSL.
JOBS=~/jobs
RUN=~/data/runs/${RUN:-m2a}
# A pid file only counts if that pid is still the process we started: pids are reused (e.g. after a
# WSL restart), so the process start time is saved next to the pid and compared.
started_at() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f20; }
alive() { [ -f "$JOBS/$1.pid" ] && [ -n "$(started_at "$(cat "$JOBS/$1.pid")")" ] && [ "$(started_at "$(cat "$JOBS/$1.pid")")" = "$(cat "$JOBS/$1.start" 2>/dev/null)" ]; }
declare -A was
for j in gen pack train m2gate m3gate rl sp ladder exhib; do was[$j]=$(alive $j && echo 1 || echo 0); done
seen_log=$(wc -l < "$RUN/log.jsonl" 2>/dev/null || echo 0)
declare -A seen_g seen_m
for g in m2gate m3gate exhib; do seen_g[$g]=$(wc -l < "$JOBS/$g.log" 2>/dev/null || echo 0); done
MATCHES=/mnt/c/Users/honza/Desktop/InfiniteChess/data/matches
declare -A seen_j
for j in rl sp ladder; do seen_j[$j]=$(wc -l < "$JOBS/$j.log" 2>/dev/null || echo 0); done
# M4 lines worth a notification: trainer every 10k steps and its forgetting check, every 10th
# self-play shard, ladder results, and any error.
FILTER_rl='"step": [0-9]*0000,|distill_eval|resumed|new run|Traceback|Error|stopped'
FILTER_sp='"shard": [0-9]*0,|self-play:|Traceback|Error|stopped'
FILTER_ladder='LADDER|failed|Traceback|Error'
for f in "$MATCHES"/{m3_,exhib_}*.jsonl; do [ -f "$f" ] && seen_m[$f]=$(( $(wc -l < "$f") / 20 )); done
while true; do
  for j in gen pack train m2gate m3gate rl sp ladder exhib; do
    now=$(alive $j && echo 1 || echo 0)
    if [ "${was[$j]}" = 1 ] && [ "$now" = 0 ]; then
      echo "[$(date +%H:%M)] JOB $j STOPPED; last log: $(tail -n 1 "$JOBS/$j.log" | tr '\r' ' ' | cut -c1-200)"
    fi
    was[$j]=$now
  done
  if [ -f "$RUN/log.jsonl" ]; then
    n=$(wc -l < "$RUN/log.jsonl")
    if [ "$n" -gt "$seen_log" ]; then
      tail -n $((n - seen_log)) "$RUN/log.jsonl" | while read -r line; do
        case $line in
          *eval*) echo "[$(date +%H:%M)] EVAL $line" ;;
          *'"step": 100,'*) echo "[$(date +%H:%M)] TRAINING STARTED $line" ;;
          *'0000, "lr"'*) echo "[$(date +%H:%M)] TRAIN $line" ;;
        esac
      done
      seen_log=$n
    fi
  fi
  for g in m2gate m3gate exhib; do
    [ -f "$JOBS/$g.log" ] || continue
    n=$(wc -l < "$JOBS/$g.log")
    if [ "$n" -gt "${seen_g[$g]:-0}" ]; then
      tail -n $((n - ${seen_g[$g]:-0})) "$JOBS/$g.log" | grep -v '^\[[0-9]*/' | sed "s/^/[$(date +%H:%M)] GATE /"
      seen_g[$g]=$n
    fi
  done
  for f in "$MATCHES"/{m3_,exhib_}*.jsonl; do
    [ -f "$f" ] || continue
    k=$(( $(wc -l < "$f") / 20 ))
    if [ "$k" -gt "${seen_m[$f]:-0}" ]; then
      echo "[$(date +%H:%M)] MATCH $(basename "$f"): $(python3 -c '
import json, sys
w = d = l = 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    a_white = r["white"].startswith(("m3", "rl-", "m2a"))
    if r["result"] == "1/2-1/2": d += 1
    elif (r["result"] == "1-0") == a_white: w += 1
    else: l += 1
n = w + d + l
print("%d games, +%d =%d -%d, score %.3f" % (n, w, d, l, (w + d / 2) / n))' "$f")"
      seen_m[$f]=$k
    fi
  done
  for j in rl sp ladder; do
    [ -f "$JOBS/$j.log" ] || continue
    n=$(wc -l < "$JOBS/$j.log")
    if [ "$n" -gt "${seen_j[$j]:-0}" ]; then
      f=FILTER_$j
      tail -n $((n - ${seen_j[$j]:-0})) "$JOBS/$j.log" | grep -E "${!f}" | cut -c1-300 | sed "s/^/[$(date +%H:%M)] ${j^^} /"
      seen_j[$j]=$n
    fi
  done
  sleep 60
done
