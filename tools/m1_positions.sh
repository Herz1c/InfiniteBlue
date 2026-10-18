#!/bin/bash
# Builds the M1 test position sets in data/m1/ (run inside WSL from the repo root).
set -e
ICORE=~/build/icore/release/icore
mkdir -p data/m1

# 1. The 24 hand-made probes (every position along their move lists).
grep -v '^#' ref/perft/positions.icn | $ICORE positions > data/m1/probes.icn

# 2. Apeiron self-play (data_gen) and 3. our M0 match games: whole games -> every position.
python3 - <<'EOF' > data/m1/games_full.icn
import glob, json, os
for path in glob.glob(os.path.expanduser("~/dg/*.jsonl")):
    for line in open(path):
        g = json.loads(line)
        print(g["start_icn"] + " " + "|".join(g["moves"]))
for line in open("data/matches/m0_L8_vs_L1.jsonl"):
    g = json.loads(line)
    print(g["start"] + " " + "|".join(g["moves"]))
EOF
$ICORE positions < data/m1/games_full.icn > data/m1/games.icn

# 4. Random playouts from the start inside the site's engine border, with far slides.
BORDER="-9223372036854773806,9223372036854773815,-9223372036854773806,9223372036854773815"
echo "w 0/100 1 (8|1) $BORDER P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|R1,1+|R8,1+|r1,8+|r8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+" \
  | $ICORE random 40 250 7 > data/m1/random.icn

wc -l data/m1/*.icn

# 5. Random playouts from every probe (sparse, tactical: pins, checks, en passant, castling).
grep -v '^#' ref/perft/positions.icn \
  | sed "s/(8|1) /(8|1) $BORDER /" \
  | { s=0; while read -r icn; do s=$((s+1)); echo "$icn" | $ICORE random 15 80 $s; done; } > data/m1/probe_random.icn
wc -l data/m1/probe_random.icn
