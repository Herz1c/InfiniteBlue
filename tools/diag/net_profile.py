# Which kernels dominate one self-play-sized forward pass? (torch profiler, CUDA time)
#   ~/venv/ic/bin/python tools/diag/net_profile.py CKPT [games] [leaves]
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
import icore  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from engine import load  # noqa: E402
from selfplay import start_icn  # noqa: E402
from train import forward, to_device  # noqa: E402

model = load(sys.argv[1])
games = int(sys.argv[2]) if len(sys.argv) > 2 else 128
leaves = int(sys.argv[3]) if len(sys.argv) > 3 else 8
g = icore.Games([start_icn()] * games)
for _ in range(5):  # grow the trees a little so leaves are varied
    enc, counts = g.select(leaves, 12)
    n = sum(counts); w = enc["cands"][1][1]
    g.backup(np.full((n, w), 1.0 / w, np.float32).tobytes(), w, np.zeros(n, np.float32).tobytes(), 12)
enc, counts = g.select(leaves, 12)
host = {k: torch.from_numpy(np.frombuffer(v, dtype=np.int32 if k in ("n_cands", "target") else np.uint8).reshape(sh).copy())
        for k, (v, sh) in enc.items()}
print({k: tuple(v.shape) for k, v in host.items()})
t = time.time(); b = to_device(host, "cuda"); torch.cuda.synchronize(); print(f"host->device {1000 * (time.time() - t):.1f} ms")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(3):
        forward(model, b)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(5):
        forward(model, b)
    torch.cuda.synchronize()
    print(f"forward {1000 * (time.time() - t) / 5:.1f} ms for {host['tokens'].shape[0]} leaves")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        forward(model, b)
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14))
