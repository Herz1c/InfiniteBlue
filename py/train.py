# Distillation training (M2): learns Apeiron's move choice (policy) and search score (value).
# Resumable: a checkpoint (model, optimizer, step) is written atomically every --ckpt-every steps
# and when the job is stopped (SIGINT/SIGTERM); starting again continues from it. Batches are drawn
# at random from whatever shards exist at that moment, so training can run while data_gen and
# pack keep adding shards.
#   python py/train.py --shards ~/data/shards/d12 --run ~/data/runs/m2a
import argparse
import json
import math
import os
import random
import signal
import time

import icore
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from net import Net, losses

K = 400.0  # score scale: target = sigmoid(score_cp / K)
VALID_EVERY = 50  # every 50th shard is held out for evaluation

stop = False


def on_signal(*_):
    global stop
    stop = True


def shard_lists(path):
    names = sorted(f for f in os.listdir(path) if f.startswith("shard_") and f.endswith(".npz") and ".tmp" not in f)
    is_valid = lambda f: int(f[6:11]) % VALID_EVERY == VALID_EVERY - 1
    return [f for f in names if not is_valid(f)], [f for f in names if is_valid(f)]


def load_shard(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def take(shard, idx):
    """Positions `idx` of a shard as python lists for icore.encode, plus value targets."""
    cut = lambda blob, off, i: bytes(blob[off[i]:off[i + 1]]).decode()
    icns = [cut(shard["icn_blob"], shard["icn_off"], i) for i in idx]
    moves = [cut(shard["move_blob"], shard["move_off"], i) for i in idx]
    sign = np.where(shard["stm"][idx] == 0, 1, -1)
    score = np.clip(shard["score"][idx] * sign, -4000, 4000)
    wdl = shard["result"][idx] * sign + 1  # side to move: 0 loss, 1 draw, 2 win
    return icns, [int(r) for r in shard["rep"][idx]], moves, score, wdl


def encode_batch(icns, reps, moves, score, wdl, mirror, max_cands, threads):
    enc = icore.encode(icns, reps, moves, mirror, max_cands, threads)
    arr = {k: torch.from_numpy(np.frombuffer(b, dtype=np.int32 if k in ("n_cands", "target") else np.uint8).reshape(s).copy())
           for k, (b, s) in enc.items()}
    arr["score_target"] = torch.sigmoid(torch.tensor(score, dtype=torch.float32) / K)
    arr["wdl"] = torch.tensor(wdl, dtype=torch.long)
    return arr


class Batches(IterableDataset):
    """Endless random batches: `per_shard` positions from each of several random shards."""

    def __init__(self, path, batch, shards_per_batch, max_cands, threads, seed):
        self.path, self.batch, self.spb = path, batch, shards_per_batch
        self.max_cands, self.threads, self.seed = max_cands, threads, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        cache, order = {}, []
        while True:
            train, _ = shard_lists(self.path)
            parts = []
            for name in rng.sample(train, min(self.spb, len(train))):
                if name not in cache:
                    cache[name] = load_shard(os.path.join(self.path, name))
                    order.append(name)
                    if len(order) > 24:
                        cache.pop(order.pop(0))
                s = cache[name]
                n = len(s["rep"])
                parts.append((s, rng.sample(range(n), min(n, self.batch // self.spb))))
            cols = [sum((list(c) for c in col), []) for col in zip(*(take(s, idx) for s, idx in parts))]
            mirror = [rng.random() < 0.5 for _ in cols[0]]
            yield encode_batch(*cols, mirror, self.max_cands, self.threads)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * step / args.warmup
    p = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def to_device(arr, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in arr.items()}


def forward(model, b):
    return model(b["tokens"].int(), b["pairs"], b["cands"].long(), b["cand_rel"], b["n_cands"].long())


@torch.no_grad()
def evaluate(model, valid, dev):
    model.eval()
    tot = {"policy": 0.0, "value": 0.0, "wdl": 0.0, "top1": 0.0, "n": 0}
    for b in valid:
        b = to_device(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward(model, b)
        _, parts = losses(out, b["target"].long(), b["wdl"], b["score_target"])
        has = b["target"] >= 0
        tot["top1"] += (out[0].argmax(1)[has] == b["target"][has]).float().mean().item()
        for k in ("policy", "value", "wdl"):
            tot[k] += parts[k]
        tot["n"] += 1
    model.train()
    return {k: v / tot["n"] for k, v in tot.items() if k != "n"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-cands", type=int, default=384)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4, help="Rust encoder threads per worker")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--snapshot-every", type=int, default=10_000, help="keep ckpt_<step>.pt for selection (PROGRESS E6)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--min-shards", type=int, default=4)
    args = ap.parse_args()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    os.makedirs(args.run, exist_ok=True)
    dev = "cuda"

    model = Net(args.d, args.layers, args.heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ckpt_path = os.path.join(args.run, "ckpt.pt")
    step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    while len(shard_lists(args.shards)[0]) < args.min_shards and not stop:
        print("waiting for shards...", flush=True)
        time.sleep(60)

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    # Fixed evaluation set: 4096 positions from the first held-out shard (once it exists).
    valid = None
    log = open(os.path.join(args.run, "log.jsonl"), "a")
    loader = DataLoader(Batches(args.shards, args.batch, 8, args.max_cands, args.threads, seed=step),
                        batch_size=None, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    t0, seen = time.time(), 0
    for b in loader:
        if stop or step >= args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        b = to_device(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward(model, b)
        loss, parts = losses(out, b["target"].long(), b["wdl"], b["score_target"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        seen += len(b["wdl"])

        if step % 100 == 0:
            rate = seen / (time.time() - t0)
            t0, seen = time.time(), 0
            rec = {"step": step, "lr": lr_at(step, args), "pos_per_s": round(rate), **{k: round(v, 4) for k, v in parts.items()}}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.eval_every == 0:
            _, valid_names = shard_lists(args.shards)
            if valid is None and valid_names:
                vs = load_shard(os.path.join(args.shards, valid_names[0]))
                idx = random.Random(0).sample(range(len(vs["rep"])), min(4096, len(vs["rep"])))
                valid = [encode_batch(*take(vs, idx[i:i + 512]), [False] * len(idx[i:i + 512]), args.max_cands, 8)
                         for i in range(0, len(idx), 512)]
            if valid:
                rec = {"step": step, "eval": evaluate(model, valid, dev)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
        if step % args.ckpt_every == 0:
            save()
        if step % args.snapshot_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, os.path.join(args.run, f"ckpt_{step:07d}.pt"))
    save()
    if step >= args.steps:
        open(os.path.join(args.run, "DONE"), "w").write(str(step))  # chains wait for this marker
    print(f"stopped at step {step}, checkpoint saved", flush=True)
    os._exit(0)  # DataLoader workers are endless generators; don't wait for them


if __name__ == "__main__":
    main()
