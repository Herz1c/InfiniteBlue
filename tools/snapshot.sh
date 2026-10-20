#!/bin/bash
# Every 30 minutes, appends one status line to PROGRESS.md section 12 ("Auto-snapshots"). Since
# 2026-10-20 the rows describe M4 (PROGRESS E12): self-play samples and speed, trainer step and
# replay ratio, the latest distillation check, the latest ladder point, and which jobs run.
# (Rows before that describe M2: data_gen games, shards, m2a training.) Run as a job:
#   bash tools/jobs.sh start snap
ROOT=/mnt/c/Users/honza/Desktop/InfiniteChess
P=$ROOT/PROGRESS.md
M4=~/data/runs/m4
INTERVAL=${INTERVAL:-1800}
grep -q '^## 12. Auto-snapshots' "$P" || printf '\n---\n\n## 12. Auto-snapshots (every 30 min, written by `tools/snapshot.sh`)\n' >> "$P"
grep -q '^### M4 rows' "$P" || printf '\n### M4 rows (from 2026-10-20)\n\n| time | self-play samples | samples/s (last shard) | trainer step | replay ratio | distill check (step: top-1 / policy) | last ladder (step: score) | jobs running |\n|---|---|---|---|---|---|---|---|\n' >> "$P"
trap "exit 0" INT TERM
# A pid file only counts if that pid is still the process we started: pids are reused (e.g. after a
# WSL restart), so the process start time is saved next to the pid and compared.
started_at() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f20; }
alive() { local f=~/jobs/$1; [ -f "$f.pid" ] && [ -n "$(started_at "$(cat "$f.pid")")" ] && [ "$(started_at "$(cat "$f.pid")")" = "$(cat "$f.start" 2>/dev/null)" ]; }
while true; do
  row=$(python3 - "$M4" <<'EOF'
import json, os, sys
m4 = sys.argv[1]
def lines(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except OSError:
        return []
log = lines(os.path.join(m4, "log.jsonl"))
train = [r for r in log if "lr" in r]
dist = [r for r in log if "distill_eval" in r]
lad = lines(os.path.join(m4, "ladder.jsonl"))
sp = [json.loads(l) for l in open(os.path.expanduser("~/jobs/sp.log")) if l.startswith('{"shard"')] if os.path.exists(os.path.expanduser("~/jobs/sp.log")) else []
t = train[-1] if train else {}
d = dist[-1] if dist else None
print(" | ".join([
    str(t.get("sp_generated", "-")),
    str(sp[-1]["samples_per_s"]) if sp else "-",
    str(t.get("step", "-")),
    str(t.get("replay_ratio", "-")),
    "%d: %.4f / %.4f" % (d["step"], d["distill_eval"]["top1"], d["distill_eval"]["policy"]) if d else "-",
    "%d: %.3f" % (lad[-1]["step"], lad[-1]["score"]) if lad else "-",
]))
EOF
)
  running=$(for j in rl sp ladder exhib m3gate pack gen train; do alive $j && printf '%s ' $j; done)
  echo "| $(date '+%m-%d %H:%M') | $row | ${running:-none} |" >> "$P"
  sleep "$INTERVAL" & wait $!  # wait (not a foreground sleep) lets the trap stop the job at once
done
