# Times the RL trainer's pieces on real self-play shards: batch building (CPU) and one
# forward+backward (GPU), plus the candidate width of the batch.
#   ~/venv/ic/bin/python tools/diag/rl_step_time.py SP_DIR INIT_CKPT
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
import multiprocessing as mp  # noqa: E402
import torch  # noqa: E402
from net import Net  # noqa: E402
from train import forward, to_device  # noqa: E402
from train_rl import Batches, rl_losses  # noqa: E402

sp, init = sys.argv[1], sys.argv[2]
args = argparse.Namespace(sp=sp, distill=os.path.expanduser("~/data/shards/d12"), batch=512, window=100000,
                          shard_samples=500, distill_start=0.3, distill_end=0.05, distill_steps=100000, threads=8, max_cands=384)
it = iter(Batches(args, mp.Value("i", 0), seed=1))
na = torch.load(init, map_location="cpu")["args"]
model = Net(na["d"], na["layers"], na["heads"]).cuda()
model.load_state_dict(torch.load(init, map_location="cuda")["model"])
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
for i in range(4):
    t = time.time(); b = next(it); t_batch = time.time() - t
    b.pop("n_sp")
    t = time.time(); b = to_device(b, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = forward(model, b)
    loss, parts = rl_losses(out, b)
    opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
    print(f"batch {t_batch:.2f} s, step {time.time() - t:.2f} s, width {b['cands'].shape[1]}, "
          f"gpu mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, {parts}", flush=True)
