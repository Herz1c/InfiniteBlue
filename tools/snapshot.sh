#!/bin/bash
# Every 30 minutes, appends one status line per job to PROGRESS.md section 12 ("Auto-snapshots"):
# data_gen games, shards, training step/throughput and the latest validation eval. Run as a job:
#   bash tools/jobs.sh start snap
ROOT=/mnt/c/Users/honza/Desktop/InfiniteChess
P=$ROOT/PROGRESS.md
RUN=~/data/runs/${RUN:-m2a}
INTERVAL=${INTERVAL:-1800}
grep -q '^## 12. Auto-snapshots' "$P" || printf '\n---\n\n## 12. Auto-snapshots (every 30 min, written by `tools/snapshot.sh`)\n\n| time | games | shards | train step | pos/s | last eval (step: top-1 / policy / value) | jobs running |\n|---|---|---|---|---|---|---|\n' >> "$P"
trap "exit 0" INT TERM
alive() { [ -f ~/jobs/$1.pid ] && kill -0 "$(cat ~/jobs/$1.pid)" 2>/dev/null; }
while true; do
  games=$(wc -l < ~/data/apeiron/classical_d12.jsonl 2>/dev/null || echo 0)
  shards=$(ls ~/data/shards/d12 2>/dev/null | grep -c '\.npz$')
  last=$(grep -v eval "$RUN/log.jsonl" 2>/dev/null | tail -n 1)
  step=$(echo "$last" | python3 -c 'import json,sys; l=sys.stdin.read().strip(); print(json.loads(l)["step"] if l else "-")')
  rate=$(echo "$last" | python3 -c 'import json,sys; l=sys.stdin.read().strip(); print(json.loads(l)["pos_per_s"] if l else "-")')
  ev=$(grep eval "$RUN/log.jsonl" 2>/dev/null | tail -n 1 | python3 -c 'import json,sys
l = sys.stdin.read().strip()
r = json.loads(l) if l else None
print("%d: %.4f / %.4f / %.4f" % (r["step"], r["eval"]["top1"], r["eval"]["policy"], r["eval"]["value"]) if r else "-")')
  running=$(for j in gen pack train m2gate; do alive $j && printf '%s ' $j; done)
  echo "| $(date '+%m-%d %H:%M') | $games | $shards | $step | $rate | $ev | ${running:-none} |" >> "$P"
  sleep "$INTERVAL" & wait $!  # wait (not a foreground sleep) lets the trap stop the job at once
done
