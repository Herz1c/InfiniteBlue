# Does torch.compile speed up self-play-sized inference? Same batch, eager vs compiled.
#   ~/venv/ic/bin/python tools/diag/compile_test.py CKPT
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
g = icore.Games([start_icn()] * 128)
batches = []
for _ in range(8):
    enc, counts = g.select(8, 12)
    n = sum(counts); w = enc["cands"][1][1]
    g.backup(np.full((n, w), 1.0 / w, np.float32).tobytes(), w, np.zeros(n, np.float32).tobytes(), 12)
    batches.append(to_device({k: torch.from_numpy(np.frombuffer(v, dtype=np.int32 if k in ("n_cands", "target") else np.uint8).reshape(sh).copy())
                              for k, (v, sh) in enc.items()}, "cuda"))


def bench(m, label):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batches[:3]:
            forward(m, b)
        torch.cuda.synchronize()
        t, n = time.time(), 0
        for _ in range(3):
            for b in batches:
                ref = forward(m, b); n += b["tokens"].shape[0]
        torch.cuda.synchronize()
    print(f"{label}: {n / (time.time() - t):.0f} leaves/s", flush=True)
    return ref


ref = bench(model, "eager")
cm = torch.compile(model, dynamic=True)
t = time.time(); bench(cm, "compiled (first run incl. compile)"); print(f"compile+run {time.time() - t:.0f} s")
out = bench(cm, "compiled")
print("max |logit diff|", float((out[0] - ref[0]).abs().max()), "max |score diff|", float((out[2] - ref[2]).abs().max()))
