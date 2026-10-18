import sys, json, time, os, torch, numpy as np
sys.path.insert(0, "/mnt/c/Users/honza/Desktop/InfiniteChess/py")
import icore
from engine import load, best_move
from train import forward, to_device
model = load(os.path.expanduser("~/data/runs/smoke/ckpt.pt")); best_move(model, "w 0/100 1 (8|1) K5,1|Q4,1|R1,1|k5,8|q4,8|r1,8")
o = json.loads(open("/mnt/c/Users/honza/Desktop/InfiniteChess/bench/openings_classical_v1.jsonl").readline())
s = icore.Search(o["start"] + " " + "|".join(o["moves"]))
T = {"select+encode": 0, "to_gpu": 0, "net": 0, "softmax+tolist": 0, "backup": 0}; leaves = 0
for it in range(150):
    t = time.time(); enc, n = s.select(64, 8); T["select+encode"] += time.time() - t
    if not n: continue
    leaves += n
    t = time.time(); b = to_device({k: torch.from_numpy(np.frombuffer(v, dtype=np.int32 if k in ("n_cands", "target") else np.uint8).reshape(sh).copy()) for k, (v, sh) in enc.items()}, "cuda"); torch.cuda.synchronize(); T["to_gpu"] += time.time() - t
    t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): logits, _, score = forward(model, b)
    torch.cuda.synchronize(); T["net"] += time.time() - t
    t = time.time(); pr = torch.softmax(logits, 1).cpu().tolist(); c = b["n_cands"].tolist(); P = [p[:k] for p, k in zip(pr, c)]; V = (2*torch.sigmoid(score)-1).cpu().tolist(); T["softmax+tolist"] += time.time() - t
    t = time.time(); s.backup(P, V); T["backup"] += time.time() - t
tot = sum(T.values()); print(f"leaves {leaves} in {tot:.2f}s = {leaves/tot:.0f}/s"); [print(f"  {k:16s} {v/tot*100:5.1f}%") for k, v in T.items()]
