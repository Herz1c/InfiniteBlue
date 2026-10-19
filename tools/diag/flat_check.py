# Exactness and speed check for the unpadded self-play path (A8): the same leaves through
# Games.select (padded, old launch) and Games.select_flat (bucketed GPU padding, new launch)
# must give the same priors and values (up to bf16 rounding).
#   ~/venv/ic/bin/python tools/diag/flat_check.py CKPT
import copy
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
import icore  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from engine import load  # noqa: E402
from selfplay import launch, load_starts  # noqa: E402
from train import forward, to_device  # noqa: E402

model = load(sys.argv[1])
starts = load_starts(os.path.expanduser("~/data/sp_starts.txt"))[:128]


def old_launch(enc):
    b = to_device({k: torch.from_numpy(np.frombuffer(v, dtype=np.int32 if k in ("n_cands", "target") else np.uint8).reshape(sh).copy())
                   for k, (v, sh) in enc.items()}, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _, score = forward(model, b)
    return torch.softmax(logits.float(), dim=1), 2 * torch.sigmoid(score.float()) - 1


# Two identical forests grown with the same (real) priors; then one batch through each path.
a, b = icore.Games(starts), icore.Games(starts)
for _ in range(6):
    enc_a, ca = a.select(8, 12)
    enc_b, cb = b.select_flat(8, 12)
    p, v = launch(model, enc_b)
    pp, vv = p.cpu().numpy(), v.cpu().numpy()
    for g in (a, b):
        g.backup(pp.tobytes(), pp.shape[1], vv.tobytes(), 12)
enc_old, c_old = a.select(8, 12)
enc_new, c_new = b.select_flat(8, 12)
assert c_old == c_new, "different leaves"
p_old, v_old = old_launch(enc_old)
p_new, v_new = launch(model, enc_new)
w = p_new.shape[1]
print("leaves", sum(c_old), "width", w, "max |prior diff|", float((p_old[:, :w] - p_new).abs().max()),
      "max |value diff|", float((v_old - v_new).abs().max()), "argmax agree", float((p_old[:, :w].argmax(1) == p_new.argmax(1)).float().mean()), flush=True)
for name in ("padded", "flat+buckets"):
    t_s = t_n = 0.0; n = 0
    for r in range(8 if name == "padded" else 25):
        t = time.time()
        enc, counts = a.select(8, 12) if name == "padded" else b.select_flat(8, 12)
        t_s += time.time() - t
        t = time.time(); p, v = old_launch(enc) if name == "padded" else launch(model, enc); torch.cuda.synchronize(); t_n += time.time() - t
        n += sum(counts)
        pp = p.cpu().numpy()
        (a if name == "padded" else b).backup(pp.tobytes(), pp.shape[1], v.cpu().numpy().tobytes(), 12)
    print(f"{name}: select {n / t_s:.0f} leaves/s, net {n / t_n:.0f} leaves/s (eager)", flush=True)
