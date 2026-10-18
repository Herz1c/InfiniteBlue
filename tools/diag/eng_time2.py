import sys, json, time, os, torch
sys.path.insert(0, "/mnt/c/Users/honza/Desktop/InfiniteChess/py")
import icore
from engine import load, best_move
model = load(os.path.expanduser("~/data/runs/smoke/ckpt.pt"))
ops = [json.loads(l) for l in open("/mnt/c/Users/honza/Desktop/InfiniteChess/bench/openings_classical_v1.jsonl")][:8]
for o in ops:
    icn = o["start"] + " " + "|".join(o["moves"])
    t = time.time(); m, s = best_move(model, icn); torch.cuda.synchronize()
    print(f"{1e3*(time.time()-t):7.1f} ms  {m}  cands={len(icore.candidates(icore.position(icn)[0]))}")
