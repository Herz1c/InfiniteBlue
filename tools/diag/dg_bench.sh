#!/bin/bash
# Wall-clock throughput of Apeiron data_gen on Classical at two depths.
cd ~/apeiron-v7.10.0
for d in 9 12; do
  out=~/dg/classical_d$d.jsonl
  rm -f $out
  s=$(date +%s.%N)
  ./target/release/data_gen --games 72 --depth $d --variants Classical --threads 18 --keep-noisy --out $out > ~/dg/log_d$d.txt 2>&1
  e=$(date +%s.%N)
  python3 - "$out" "$s" "$e" "$d" <<'EOF'
import json, sys
path, s, e, d = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
g = [json.loads(l) for l in open(path)]
pos = [p for x in g for p in x["positions"]]
wall = e - s
res = {}
for x in g: res[x["termination"]] = res.get(x["termination"], 0) + 1
print(f"depth {d}: games={len(g)} positions={len(pos)} wall={wall:.0f}s "
      f"-> {len(pos)/wall*3600:,.0f} positions/h, mean ms/move={sum(p['ms'] for p in pos)/len(pos):.1f}, "
      f"mean nodes={sum(p['nodes'] for p in pos)/len(pos):,.0f}, mean depth={sum(p['depth'] for p in pos)/len(pos):.1f}")
print("  results:", {r: sum(1 for x in g if x['result']==r) for r in ('1-0','0-1','1/2-1/2')}, "terminations:", res)
pl = sorted(x["plies"] for x in g); print("  plies median", pl[len(pl)//2], "max", pl[-1])
EOF
done
