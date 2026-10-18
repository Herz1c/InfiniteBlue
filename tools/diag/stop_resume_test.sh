#!/bin/bash
# Plumbing test for stopping and resuming a gate match exactly the way `jobs.sh stop` does
# (setsid job, then SIGINT to its whole process group). Dev suite and scratch files only.
ROOT=/mnt/c/Users/honza/Desktop/InfiniteChess
PY=~/venv/ic/bin/python
S=~/scratch/stoptest; rm -rf "$S"; mkdir -p "$S"
cat > "$S/chain.sh" <<EOT
trap 'echo "chain: stopped by request"; exit 130' INT TERM
cd $ROOT/py
$PY match.py --a "$PY $ROOT/py/engine.py --ckpt $HOME/data/runs/m2a/ckpt_0200000.pt --mode search" \
  --a-name m3-mcts --b ~/engines/apeiron-v7.10.0/sprt --b-strength 4 --openings ../bench/openings_dev_v1.jsonl \
  --pairs 40 --ms 1000 --concurrency 4 --out $S/games.jsonl | tail -n 3
[ "\${PIPESTATUS[0]}" = 130 ] && { echo "chain: stopped by request"; exit 130; }
echo "chain: finished"
EOT
start() { setsid nohup bash "$S/chain.sh" >> "$S/chain.log" 2>&1 < /dev/null & echo $! > "$S/pid"; }
start; sleep 45
pid=$(cat "$S/pid")
echo "pgid of job: $(ps -o pgid= -p $pid)  (pid $pid)"
echo "games before stop: $(wc -l < $S/games.jsonl 2>/dev/null || echo 0)"
kill -INT -- "-$pid"; sleep 8
echo "--- chain log:"; cat "$S/chain.log"
echo "left over engine/match processes: $(pgrep -fc 'engine.py --ckpt|apeiron-v7.10.0/sprt|match.py')"
echo "games after stop: $(wc -l < $S/games.jsonl), engine failures: $(grep -c 'Engine failure' $S/games.jsonl)"
start; sleep 40; kill -INT -- "-$(cat $S/pid)"; sleep 8
echo "--- chain log after resume + second stop:"; tail -n 3 "$S/chain.log"
echo "games: $(wc -l < $S/games.jsonl); duplicates: $(python3 -c "
import json; k=[(json.loads(l)['pair'],json.loads(l)['game']) for l in open('$S/games.jsonl')]; print(len(k)-len(set(k)))"); failures: $(grep -c 'Engine failure' $S/games.jsonl)"
echo "left over processes: $(pgrep -fc 'engine.py --ckpt|apeiron-v7.10.0/sprt|match.py')"
