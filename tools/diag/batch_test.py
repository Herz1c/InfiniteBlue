import sys, json, time, os
sys.path.insert(0, "/mnt/c/Users/honza/Desktop/InfiniteChess/py")
from engine import load, best_move, search_move
model = load(os.path.expanduser("~/data/runs/smoke/ckpt.pt")); best_move(model, "w 0/100 1 (8|1) K5,1|Q4,1|R1,1|k5,8|q4,8|r1,8")
ops = [json.loads(l) for l in open("/mnt/c/Users/honza/Desktop/InfiniteChess/bench/openings_classical_v1.jsonl")][:3]
for batch in (32, 64, 128, 256):
    tot = 0
    for o in ops:
        m, sc, v = search_move(model, o["start"] + " " + "|".join(o["moves"]), 3000, batch=batch); tot += v
    print(f"batch {batch:3d}: {tot/ (3*2.7):.0f} visits/s")
