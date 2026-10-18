import sys, json, time, os, torch
sys.path.insert(0, "/mnt/c/Users/honza/Desktop/InfiniteChess/py")
from engine import load, best_move, search_move
model = load(os.path.expanduser("~/data/runs/smoke/ckpt.pt"))
best_move(model, "w 0/100 1 (8|1) K5,1|Q4,1|R1,1|k5,8|q4,8|r1,8")
ops = [json.loads(l) for l in open("/mnt/c/Users/honza/Desktop/InfiniteChess/bench/openings_classical_v1.jsonl")][:3]
for o in ops:
    icn = o["start"] + " " + "|".join(o["moves"])
    for ms in (1000, 5000):
        t = time.time(); m, sc, v = search_move(model, icn, ms)
        print(f"{ms} ms budget: {time.time()-t:.2f}s, visits {v} ({v/(time.time()-t):.0f}/s), move {m}, score {sc:.0f}; policy says {best_move(model, icn)[0]}")
