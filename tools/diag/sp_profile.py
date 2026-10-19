# Where does self-play time go? Times select (CPU), the net (GPU) and backup (CPU) for one group
# of games, without pipelining.
#   ~/venv/ic/bin/python tools/diag/sp_profile.py CKPT [games] [leaves] [threads]
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
import icore  # noqa: E402
import torch  # noqa: E402
from engine import load  # noqa: E402
from selfplay import finish, launch, start_icn  # noqa: E402

ckpt = sys.argv[1]
games = int(sys.argv[2]) if len(sys.argv) > 2 else 128
leaves = int(sys.argv[3]) if len(sys.argv) > 3 else 8
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 12
model = load(ckpt)
g = icore.Games([start_icn()] * games)
t = {"select": 0.0, "net": 0.0, "backup": 0.0}
n = 0
widths = []
for r in range(60):
    a = time.time(); enc, counts = g.select(leaves, threads); b = time.time()
    p = launch(model, enc)
    if p is not None:
        torch.cuda.synchronize()
    c = time.time()
    finish(g, p, threads)
    d = time.time()
    if r >= 10:  # skip warm-up rounds
        t["select"] += b - a; t["net"] += c - b; t["backup"] += d - c; n += sum(counts)
        widths.append(enc["cands"][1][1])
total = sum(t.values())
print(f"games {games} leaves {leaves} threads {threads}: {n / total:.0f} leaves/s; "
      + ", ".join(f"{k} {v / total:.0%}" for k, v in t.items()) + f"; mean batch {n / 50:.0f}, mean width {sum(widths) / len(widths):.0f}")
