#!/bin/bash
# Long-running jobs that can be stopped and resumed at any time (run inside WSL).
#   bash tools/jobs.sh start <job>    starts it in the background, resuming where it stopped
#   bash tools/jobs.sh stop  <job>    asks it to finish cleanly (SIGINT), kills after 120 s
#   bash tools/jobs.sh status         what is running, progress, last log lines
#   bash tools/jobs.sh stop all / start all   pause / resume everything
# Jobs: gen (Apeiron self-play data), pack (games -> training shards), train (the net),
#       m2gate (unattended M2 gate chain: waits for training, selects, plays the matches),
#       m3gate (unattended M3 gate chain: waits for gen to finish, then the MCTS matches),
#       snap (appends a status line to PROGRESS.md section 12 every 30 min),
#       M4 (PROGRESS E12): rl (trainer, exports the net), sp (self-play games), ladder (snapshots vs Apeiron L8, dev suite).
ROOT=/mnt/c/Users/honza/Desktop/InfiniteChess
JOBS=~/jobs
DATA=~/data
PY=~/venv/ic/bin/python
mkdir -p "$JOBS" "$DATA/apeiron" "$DATA/shards" "$DATA/runs"

# Each job resumes on its own: data_gen appends to its output and counts the games already
# there; pack skips finished shards; train loads its last checkpoint.
run_gen() {
  exec ~/engines/apeiron-v7.10.0/data_gen --variants Classical --depth 12 --keep-noisy \
    --games 250000 --threads 16 --out "$DATA/apeiron/classical_d12.jsonl"
}
run_pack() {
  exec "$PY" "$ROOT/py/pack.py" --games "$DATA/apeiron/classical_d12.jsonl" --out "$DATA/shards/d12" --follow
}
run_train() {
  # Waits until 60 shards (~2.7M positions) exist, so the first epochs do not overfit a few shards.
  cd "$ROOT/py" && exec "$PY" train.py --shards "$DATA/shards/d12" --run "$DATA/runs/${RUN:-m2a}"     --min-shards 60 --workers 3 --threads 3
}

run_snap() {
  exec bash "$ROOT/tools/snapshot.sh"
}

run_m2gate() {
  exec bash "$ROOT/tools/chain_m2.sh"
}

run_m3gate() {
  exec bash "$ROOT/tools/chain_m3.sh"
}

M4=$DATA/runs/m4
run_rl() {
  cd "$ROOT/py" && exec "$PY" train_rl.py --run "$M4" --sp "$DATA/sp" --distill "$DATA/shards/d12"     --init "$DATA/runs/m2a/ckpt_0200000.pt" --workers 2 --threads 4 --snapshot-every 5000
}
run_sp() {
  # The trainer exports the first net when it starts; self-play waits for it.
  until [ -f "$M4/current.pt" ]; do sleep 10; done
  # One inductor compile thread: the default pool keeps a ~0.6 GB worker process per thread (RAM cap, 2026-10-19).
  cd "$ROOT/py" && TORCHINDUCTOR_COMPILE_THREADS=1 exec "$PY" selfplay.py --net "$M4/current.pt" --out "$DATA/sp" --starts "$DATA/sp_starts.txt"
}
run_ladder() {
  exec bash "$ROOT/tools/ladder_m4.sh"
}
run_exhib() {
  exec bash "$ROOT/tools/exhib_l8.sh"
}

progress() {
  case $1 in
    gen) [ -f "$DATA/apeiron/classical_d12.jsonl" ] && echo "  games: $(wc -l < "$DATA/apeiron/classical_d12.jsonl") / 250000" ;;
    pack) echo "  shards: $(ls "$DATA/shards/d12" 2>/dev/null | grep -c '\.npz$')" ;;
    train) [ -f "$DATA/runs/${RUN:-m2a}/log.jsonl" ] && tail -n 1 "$DATA/runs/${RUN:-m2a}/log.jsonl" | sed "s/^/  last: /" ;;
    rl) [ -f "$M4/log.jsonl" ] && tail -n 1 "$M4/log.jsonl" | sed "s/^/  last: /" ;;
    sp) echo "  self-play shards: $(ls "$DATA/sp" 2>/dev/null | grep -c '^sp_.*\.npz$')" ;;
    ladder) [ -f "$M4/ladder.jsonl" ] && tail -n 1 "$M4/ladder.jsonl" | sed "s/^/  last: /" ;;
  esac
}

# A pid file only counts if that pid is still the process we started: pids are reused (e.g. after a
# WSL restart), so the process start time is saved next to the pid and compared.
started_at() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f20; }
alive() { [ -f "$JOBS/$1.pid" ] && [ -n "$(started_at "$(cat "$JOBS/$1.pid")")" ] && [ "$(started_at "$(cat "$JOBS/$1.pid")")" = "$(cat "$JOBS/$1.start" 2>/dev/null)" ]; }

# "all" pauses or resumes everything (stop order: chain first, data producers last).
if [ "${2:-}" = all ]; then
  case $1 in
    stop) for j in exhib ladder sp rl m3gate m2gate snap train pack gen; do bash "$0" stop "$j"; done; exit 0 ;;
    start) for j in gen pack train m2gate m3gate rl sp ladder snap; do bash "$0" start "$j"; done; exit 0 ;;
  esac
fi

case $1 in
  run) "run_$2" ;;
  start)
    if alive "$2"; then echo "$2 is already running (pid $(cat "$JOBS/$2.pid"))"; exit 0; fi
    # setsid: own session, so closing this shell or the WSL window does not stop the job.
    setsid nohup bash "$0" run "$2" >> "$JOBS/$2.log" 2>&1 < /dev/null &
    echo $! > "$JOBS/$2.pid"
    sleep 0.2; started_at $! > "$JOBS/$2.start"
    echo "started $2 (pid $!), log: $JOBS/$2.log"
    ;;
  stop)
    if ! alive "$2"; then echo "$2 is not running"; exit 0; fi
    pid=$(cat "$JOBS/$2.pid")
    # snap only sleeps and appends lines, so it is simply terminated; the others stop gracefully on SIGINT.
    # The bash-script jobs run match.py and engines as child processes, so the whole process group
    # (setsid made the job its own group) gets the signal; match.py resumes from its output file.
    # They get SIGTERM, not SIGINT: a job started in the background inherits SIGINT as ignored, and
    # bash cannot trap a signal ignored at startup (match.py handles both alike).
    case $2 in
      snap) kill -TERM "$pid" ;;
      m2gate|m3gate|ladder|exhib) kill -TERM -- "-$pid" ;;
      *) kill -INT "$pid" ;;
    esac
    for _ in $(seq 120); do alive "$2" || break; sleep 1; done
    if alive "$2"; then case $2 in m2gate|m3gate|ladder|exhib) kill -TERM -- "-$pid" ;; *) kill -TERM "$pid" ;; esac; echo "$2 did not stop in 120 s, sent SIGTERM"; else echo "stopped $2"; fi
    rm -f "$JOBS/$2.pid" "$JOBS/$2.start"
    ;;
  status)
    for j in gen pack train m2gate m3gate rl sp ladder exhib snap; do
      if alive "$j"; then echo "$j: RUNNING (pid $(cat "$JOBS/$j.pid"))"; else echo "$j: stopped"; fi
      progress "$j"
      [ -f "$JOBS/$j.log" ] && tail -n 2 "$JOBS/$j.log" | tr '\r' '\n' | tail -n 2 | sed 's/^/  | /'
    done
    true
    ;;
  *) echo "usage: bash tools/jobs.sh start|stop <gen|pack|train|m2gate|snap|all> | status"; exit 2 ;;
esac
