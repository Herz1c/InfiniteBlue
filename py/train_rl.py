# Reinforcement-learning trainer (M4, PROGRESS E12). Trains the net on our own self-play games
# (py/selfplay.py writes sp_*.npz) plus a decaying share of Apeiron-distillation positions, and
# exports the net for self-play every --export-every steps.
#   Policy: cross-entropy to the search's visit distribution (one-hot for distillation rows); at most
#   --max-cands candidates per position: every visited move plus evenly spread unvisited ones (A7).
#   Value: sigmoid-space target (1 + 0.5 z + 0.5 q) / 2 for self-play rows; WDL target z.
#   Replay window: batches come from the newest --window self-play positions.
#   Replay ratio: the trainer waits while it has used more than --replay-ratio times the number of
#   self-play positions generated, so it cannot overfit a few games.
# Resumable like train.py: ckpt.pt (model, optimizer, step, counters) on every export and on stop.
#   python py/train_rl.py --run ~/data/runs/m4 --sp ~/data/sp --distill ~/data/shards/d12 \
#       --init ~/data/runs/m2a/ckpt_0200000.pt
import argparse
import json
import multiprocessing as mp
import os
import random
import signal
import time

import icore
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from net import Net
from train import K, evaluate, encode_batch, forward, load_shard, shard_lists, take, to_device

stop = False


def on_signal(*_):
    global stop
    stop = True


def sp_shards(path):
    return sorted(f for f in os.listdir(path) if f.startswith("sp_") and f.endswith(".npz") and ".tmp" not in f)


def take_sp(shard, idx):
    cut = lambda blob, off, i: bytes(blob[off[i]:off[i + 1]]).decode()
    icns = [cut(shard["icn_blob"], shard["icn_off"], i) for i in idx]
    pols = [shard["pol"][shard["pol_off"][i]:shard["pol_off"][i + 1]].astype(np.float32) for i in idx]
    return icns, [int(r) for r in shard["rep"][idx]], pols, shard["q"][idx], shard["z"][idx]


def keep_list(pol, cap):
    """Candidate indices kept for training (A7): every visited move, plus evenly spread unvisited
    ones up to `cap` in total (or the `cap` most visited if more were visited)."""
    visited = [i for i, x in enumerate(pol) if x > 0]
    if len(visited) >= cap:
        return sorted(sorted(visited, key=lambda i: -pol[i])[:cap])
    rest = [i for i, x in enumerate(pol) if x <= 0]
    m = min(len(rest), cap - len(visited))
    return sorted(visited + [rest[(j * len(rest)) // m] for j in range(m)])


def tensors(enc):
    return {k: torch.from_numpy(np.frombuffer(v, dtype=np.int32 if k in ("n_cands", "target") else np.uint8).reshape(sh).copy())
            for k, (v, sh) in enc.items()}


def concat(a, b):
    """Two encoded batches as one, padding the candidate dimension to the wider one."""
    c = max(a["cands"].shape[1], b["cands"].shape[1])
    pad = lambda t: F.pad(t, (0, 0) * (t.dim() - 2) + (0, c - t.shape[1])) if t.dim() > 1 and t.shape[1] < c else t
    out = {}
    for k in a:
        if k in ("cands", "cand_rel"):
            out[k] = torch.cat([pad(a[k]), pad(b[k])])
        else:
            out[k] = torch.cat([a[k], b[k]])
    return out


def distill_share(step, args):
    p = min(1.0, step / max(1, args.distill_steps))
    return args.distill_start + (args.distill_end - args.distill_start) * p


class Batches(IterableDataset):
    """Endless batches: self-play rows from random shards of the replay window, plus distillation rows."""

    def __init__(self, args, step_value, seed):
        self.args, self.step_value, self.seed = args, step_value, seed

    def __iter__(self):
        a = self.args
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        cache, order = {}, []

        def get(path):
            if path not in cache:
                cache[path] = load_shard(path)
                order.append(path)
                if len(order) > 8:  # each worker keeps at most 8 shards in RAM
                    cache.pop(order.pop(0))
            return cache[path]

        while True:
            n_d = round(a.batch * distill_share(self.step_value.value, a))
            n_sp = a.batch - n_d
            names = sp_shards(a.sp)
            window = names[-max(1, a.window // a.shard_samples):]
            icns, reps, keeps, soft_rows, score, wdl = [], [], [], [], [], []
            for name in rng.sample(window, min(8, len(window))):
                s = get(os.path.join(a.sp, name))
                idx = rng.sample(range(len(s["z"])), min(len(s["z"]), n_sp // min(8, len(window))))
                i, r, p, q, z = take_sp(s, idx)
                for pol in p:
                    k = keep_list(pol, a.max_cands)
                    keeps.append(k)
                    soft_rows.append(pol[k] / max(1e-9, pol[k].sum()))
                icns += i; reps += r
                score += list((1 + 0.5 * z + 0.5 * q) / 2)
                wdl += list(z.astype(np.int64) + 1)
            n_sp_rows = len(icns)
            b = tensors(icore.encode_keep(icns, reps, keeps, [rng.random() < 0.5 for _ in icns], a.threads))
            soft = torch.zeros(n_sp_rows, b["cands"].shape[1])
            for row, t in enumerate(soft_rows):
                if len(t) == int(b["n_cands"][row]):  # always true unless the rules changed
                    soft[row, : len(t)] = torch.from_numpy(t)
            if n_d:
                train, _ = shard_lists(a.distill)
                per = max(1, n_d // 4)
                d_icns, d_reps, d_moves = [], [], []
                for name in rng.sample(train, 4):
                    s = get(os.path.join(a.distill, name))
                    i, r, m, sc, w = take(s, rng.sample(range(len(s["rep"])), per))
                    d_icns += i; d_reps += r; d_moves += m
                    score += list(1 / (1 + np.exp(-np.asarray(sc, np.float64) / K)))
                    wdl += list(w)
                d = tensors(icore.encode(d_icns, d_reps, d_moves, [rng.random() < 0.5 for _ in d_icns], a.max_cands, a.threads))
                d_soft = torch.zeros(len(d_icns), d["cands"].shape[1])
                has = d["target"] >= 0
                d_soft[has, d["target"][has].long()] = 1.0
                b = concat(b, d)
                c = b["cands"].shape[1]
                soft = torch.cat([F.pad(soft, (0, c - soft.shape[1])), F.pad(d_soft, (0, c - d_soft.shape[1]))])
            b["soft"] = soft
            b["score_target"] = torch.tensor(score, dtype=torch.float32)
            b["wdl"] = torch.tensor(wdl, dtype=torch.long)
            b["n_sp"] = torch.tensor(n_sp_rows)
            yield b


def rl_losses(out, b, wdl_weight=0.3):
    logits, wdl_logits, score_logit = out
    has = b["soft"].sum(1) > 0
    logp = F.log_softmax(logits, dim=1)
    pol = -(b["soft"] * logp).sum(1)[has].mean() if has.any() else logits.sum() * 0
    val = F.binary_cross_entropy_with_logits(score_logit, b["score_target"])
    res = F.cross_entropy(wdl_logits, b["wdl"])
    return pol + val + wdl_weight * res, {"policy": pol.item(), "value": val.item(), "wdl": res.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sp", required=True, help="self-play shards (py/selfplay.py --out)")
    ap.add_argument("--distill", required=True, help="Apeiron distillation shards (py/pack.py --out)")
    ap.add_argument("--init", required=True, help="checkpoint to start from (used only if the run is new)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--window", type=int, default=1_500_000)
    ap.add_argument("--shard-samples", type=int, default=20000, help="self-play shard size (window in shards)")
    ap.add_argument("--distill-start", type=float, default=0.3)
    ap.add_argument("--distill-end", type=float, default=0.05)
    ap.add_argument("--distill-steps", type=int, default=100_000)
    ap.add_argument("--replay-ratio", type=float, default=4.0)
    ap.add_argument("--min-sp", type=int, default=100_000, help="self-play positions before training starts")
    ap.add_argument("--export-every", type=int, default=1000)
    ap.add_argument("--snapshot-every", type=int, default=10_000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--max-cands", type=int, default=384, help="candidates kept per training position (A7)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    os.makedirs(args.run, exist_ok=True)
    os.makedirs(args.sp, exist_ok=True)
    dev = "cuda"

    ckpt_path, export_path = os.path.join(args.run, "ckpt.pt"), os.path.join(args.run, "current.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        net_args = ck["net_args"]
    else:
        ck = None
        net_args = torch.load(args.init, map_location="cpu")["args"]
    model = Net(net_args["d"], net_args["layers"], net_args["heads"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step, used_sp = 0, 0
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, used_sp = ck["step"], ck["used_sp"]
        print(f"resumed at step {step}", flush=True)
    else:
        model.load_state_dict(torch.load(args.init, map_location=dev)["model"])
        print(f"new run from {args.init}", flush=True)

    def export(path):
        torch.save({"model": model.state_dict(), "step": step, "args": net_args}, path + ".tmp")
        os.replace(path + ".tmp", path)

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "used_sp": used_sp,
                    "net_args": net_args, "args": vars(args)}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    if not os.path.exists(export_path):
        export(export_path)  # self-play starts with the initial net

    sizes = {}

    def generated():
        for name in sp_shards(args.sp):
            if name not in sizes:
                with np.load(os.path.join(args.sp, name)) as z:
                    sizes[name] = len(z["z"])
        return sum(sizes.values())

    while generated() < args.min_sp and not stop:
        print(f"waiting for self-play positions: {generated()} / {args.min_sp}", flush=True)
        time.sleep(60)

    # Fixed distillation validation set (watch for forgetting what Apeiron taught).
    _, valid_names = shard_lists(args.distill)
    vs = load_shard(os.path.join(args.distill, valid_names[0]))
    idx = random.Random(0).sample(range(len(vs["rep"])), 4096)
    valid = [encode_batch(*take(vs, idx[i:i + 512]), [False] * 512, 384, 8) for i in range(0, 4096, 512)]

    step_value = mp.Value("i", step)
    loader = DataLoader(Batches(args, step_value, seed=step + 1), batch_size=None, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True)
    log = open(os.path.join(args.run, "log.jsonl"), "a")
    t0, seen = time.time(), 0
    for b in loader:
        if stop:
            break
        while used_sp > args.replay_ratio * generated() and not stop:
            time.sleep(20)  # wait for self-play to catch up
        if stop:
            break
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        n_sp = int(b.pop("n_sp"))
        b = to_device(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward(model, b)
        loss, parts = rl_losses(out, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        step_value.value = step
        used_sp += n_sp
        seen += len(b["wdl"])

        if step % 100 == 0:
            gen = generated()
            rec = {"step": step, "lr": lr, "pos_per_s": round(seen / (time.time() - t0)), "sp_generated": gen,
                   "replay_ratio": round(used_sp / max(1, gen), 2), "distill_share": round(distill_share(step, args), 3),
                   **{k: round(v, 4) for k, v in parts.items()}}
            t0, seen = time.time(), 0
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.eval_every == 0:
            rec = {"step": step, "distill_eval": evaluate(model, valid, dev)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.export_every == 0:
            export(export_path)
            save()
        if step % args.snapshot_every == 0:
            export(os.path.join(args.run, f"ckpt_{step:07d}.pt"))
    save()
    print(f"stopped at step {step}, checkpoint saved", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
