# Self-play for reinforcement learning (M4, PROGRESS E12). Many games run at once: Rust keeps one
# search tree per game (icore.Games) and every round collects a few leaves from every tree into
# one batch for the net. Two groups of games alternate, so the CPU walks one group's trees while
# the GPU evaluates the other's leaves.
#   Playout cap randomization: a move gets a full search (600 visits, root noise, recorded as a
#   training sample) with probability 0.25, otherwise a fast search (100 visits, not recorded).
#   Games start from --starts (Apeiron's random 12-ply openings, amendment A5). Moves are sampled
#   by visit count for the first 30 plies, then the most visited is played.
#   Finished samples are written as shards sp_NNNNNN.npz (atomic); the net file is reloaded when
#   the trainer exports a new one. Stopping (SIGINT/SIGTERM) writes the finished samples and exits;
#   games in progress are dropped.
#   python py/selfplay.py --net ~/data/runs/m4/current.pt --out ~/data/sp
import argparse
import json
import os
import random
import signal
import time

import icore
import numpy as np
import torch
import warnings

from engine import load
from train import forward, to_device

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
warnings.filterwarnings("ignore", message="The given buffer is not writable")  # torch.frombuffer on bytes; read only

stop = False


def on_signal(*_):
    global stop
    stop = True


def start_icn():
    """The Classical starting position (taken from the dev suite's start field; no opening moves)."""
    return json.loads(open(os.path.join(ROOT, "bench", "openings_dev_v1.jsonl")).readline())["start"]


def load_starts(path):
    """Starting positions, one ICN per line (tools/make_sp_starts.py); just the start position without a file."""
    if not path:
        return [start_icn()]
    return [line.strip() for line in open(path) if line.strip()]


def blob(strings):
    data = [s.encode() for s in strings]
    off = np.zeros(len(data) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(d) for d in data])
    return np.frombuffer(b"".join(data), dtype=np.uint8), off


def launch(model, enc, chunk=256):
    """Starts the net on a batch from Games.select_flat (asynchronously). Leaves are sorted by their
    number of candidates and run in chunks, each padded on the GPU only to its own widest leaf, so
    a few leaves with many candidates do not widen the whole batch. None for an empty batch."""
    n = enc["n_cands"][1][0]
    if n == 0:
        return None
    # Straight from the bytes Rust returned to the GPU, without an extra host copy.
    gpu = lambda k: torch.frombuffer(enc[k][0], dtype=torch.int32 if k == "n_cands" else torch.uint8).view(enc[k][1]).cuda()
    tokens, pairs, cands, rel, nc_g = gpu("tokens"), gpu("pairs"), gpu("cands"), gpu("cand_rel"), gpu("n_cands")
    nc = np.frombuffer(enc["n_cands"][0], dtype=np.int32)
    off = torch.from_numpy(np.concatenate([[0], np.cumsum(nc)[:-1]]).astype(np.int64)).cuda(non_blocking=True)
    order = np.argsort(nc, kind="stable")
    priors = torch.zeros(n, int(nc.max()), device="cuda")
    values = torch.empty(n, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for a in range(0, n, chunk):
            rows = order[a:a + chunk]
            cmax = max(1, int(nc[rows[-1]]))
            idx = torch.from_numpy(rows.astype(np.int64)).cuda(non_blocking=True)
            k = nc_g[idx].long()
            # Padding rows repeat the leaf's last candidate; the net masks them (candidates never
            # attend to each other, so their content does not matter).
            gi = off[idx][:, None] + torch.minimum(torch.arange(cmax, device="cuda")[None, :], (k - 1).clamp(min=0)[:, None])
            b = {"tokens": tokens[idx], "pairs": pairs[idx], "cands": cands[gi], "cand_rel": rel[gi], "n_cands": nc_g[idx]}
            logits, _, score = forward(model, b)
            priors[idx, :cmax] = torch.softmax(logits.float(), dim=1)
            values[idx] = 2 * torch.sigmoid(score.float()) - 1
    # Copy the results back asynchronously and mark when they are there, so the next batch can be
    # queued on the GPU before we wait for this one (finish waits only for this event).
    p_cpu = torch.empty(priors.shape, pin_memory=True)
    v_cpu = torch.empty(values.shape, pin_memory=True)
    p_cpu.copy_(priors, non_blocking=True)
    v_cpu.copy_(values, non_blocking=True)
    done = torch.cuda.Event()
    done.record()
    return p_cpu, v_cpu, done


def finish(games, pending, threads):
    if pending is None:
        games.backup(b"", 1, b"", threads)
        return
    priors, values, done = pending
    done.synchronize()
    p = priors.numpy()
    games.backup(p.tobytes(), p.shape[1], values.numpy().tobytes(), threads)


class Game:
    """Python-side state of one game; the tree itself lives in icore.Games."""

    def __init__(self, rng, args):
        self.rng, self.args = rng, args
        self.ply, self.moves, self.samples = 0, [], []
        self.resign_ok = rng.random() >= args.no_resign
        self.low = [0, 0]
        self.new_move(0)

    def new_move(self, visits_now):
        self.full = self.rng.random() < self.args.p_full
        self.target = self.args.full if self.full else self.args.fast
        # Visits reused from the previous move may cover at most half of the target.
        self.need = self.target - min(visits_now, self.target // 2)
        self.start_visits = visits_now
        self.noised = False


def step(games, state, args, rng, out):
    """Plays a move in every game whose search is done; finished games go to `out` and restart."""
    for g, st in enumerate(state):
        v = games.visits(g)
        if st.full and not st.noised and v > 0:
            moves, _, _ = games.root(g)
            if moves:
                games.noise(g, 0.25, min(1.0, 10.0 / len(moves)), rng.getrandbits(63))
                st.noised = True
        if v - st.start_visits < st.need:
            continue
        moves, n, q = games.root(g)
        stm = st.ply % 2  # 0: the side that moved first in this game (White from the start position)
        if st.ply < args.temp_plies and sum(n) > 0:
            mv = rng.choices(moves, weights=n)[0]
        else:
            mv = moves[max(range(len(n)), key=n.__getitem__)]
        if st.full and sum(n) > 0:
            icn, rep = games.position(g)
            total = float(sum(n))
            st.samples.append((icn, rep, [x / total for x in n], q, stm, st.ply))
        st.low[stm] = st.low[stm] + 1 if q < args.resign else 0
        if st.resign_ok and st.low[stm] >= 3:
            end_game(games, g, st, out, rng, args, winner=1 - stm, how="resign")
            continue
        games.play(g, mv)
        st.moves.append(mv)
        st.ply += 1
        end = games.game_end(g)
        if end is not None:
            res = RESULT[end[0]]
            # From the start position White moves first, so White is side 0.
            end_game(games, g, st, out, rng, args, winner=None if res == 0 else (0 if res == 1 else 1), how=end[1])
        elif st.ply >= args.max_plies:
            end_game(games, g, st, out, rng, args, winner=None, how="max plies")
        else:
            st.new_move(games.visits(g))


def end_game(games, g, st, out, rng, args, winner, how):
    for icn, rep, pol, q, stm, ply in st.samples:
        z = 0 if winner is None else (1 if winner == stm else -1)
        out["samples"].append((icn, rep, pol, q, z, ply))
    out["games"].append({"plies": st.ply, "winner": winner, "how": how, "resign_ok": st.resign_ok,
                         "samples": len(st.samples), "moves": st.moves})
    games.reset(g, rng.choice(out["starts"]))
    state_new = Game(rng, args)
    out["replace"][g] = state_new


def write_shard(path, samples, net_step):
    icn_blob, icn_off = blob([s[0] for s in samples])
    pols = [np.asarray(s[2], dtype=np.float16) for s in samples]
    pol_off = np.zeros(len(pols) + 1, dtype=np.int64)
    pol_off[1:] = np.cumsum([len(p) for p in pols])
    np.savez(path + ".tmp.npz", icn_blob=icn_blob, icn_off=icn_off,
             rep=np.array([s[1] for s in samples], np.uint8),
             pol=np.concatenate(pols) if pols else np.zeros(0, np.float16), pol_off=pol_off,
             q=np.array([s[3] for s in samples], np.float32), z=np.array([s[4] for s in samples], np.int8),
             ply=np.array([s[5] for s in samples], np.int16), net_step=np.int32(net_step))
    os.replace(path + ".tmp.npz", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="net file; reloaded whenever it changes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=128, help="games per group (two groups)")
    ap.add_argument("--leaves", type=int, default=8, help="leaves per tree per round")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--full", type=int, default=600)
    ap.add_argument("--fast", type=int, default=100)
    ap.add_argument("--p-full", type=float, default=0.25)
    ap.add_argument("--temp-plies", type=int, default=30)
    ap.add_argument("--max-plies", type=int, default=600)
    ap.add_argument("--resign", type=float, default=-0.95)
    ap.add_argument("--no-resign", type=float, default=0.2, help="share of games that never resign")
    ap.add_argument("--shard-samples", type=int, default=20000)
    ap.add_argument("--max-shards", type=int, default=0, help="stop after this many shards (0 = never; tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--starts", default="", help="file of starting ICNs (amendment A5); default: the start position")
    ap.add_argument("--no-compile", action="store_true", help="skip torch.compile (about 2x slower inference)")
    ap.add_argument("--draw-penalty", type=float, default=0.1, help="both sides score moving into a draw as -this (A6)")
    args = ap.parse_args()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed or time.time_ns())

    model = load(args.net)
    net_step = torch.load(args.net, map_location="cpu").get("step", 0)
    net_mtime = os.path.getmtime(args.net)
    if not args.no_compile:
        model = torch.compile(model, dynamic=True)  # fuses the many small kernels: about 2x (tools/diag/compile_test.py)

    def reload():
        """New weights from the trainer, copied into the same (compiled) model."""
        ck = torch.load(args.net, map_location="cuda")
        getattr(model, "_orig_mod", model).load_state_dict(ck["model"])
        return ck.get("step", 0), os.path.getmtime(args.net)

    shard = len([f for f in os.listdir(args.out) if f.startswith("sp_") and f.endswith(".npz") and ".tmp" not in f])
    starts = load_starts(args.starts)
    groups = [icore.Games([rng.choice(starts) for _ in range(args.games)]) for _ in range(2)]
    for grp in groups:
        grp.set_draw_penalty(args.draw_penalty)
    states = [[Game(rng, args) for _ in range(args.games)] for _ in range(2)]
    out = {"samples": [], "games": [], "starts": starts, "replace": {}}
    log = open(os.path.join(args.out, "games.jsonl"), "a")
    print(f"self-play: net step {net_step}, {len(starts)} starting positions, starting at shard {shard}", flush=True)

    t0, leaves, written = time.time(), 0, 0
    enc, counts = groups[0].select_flat(args.leaves, args.threads)
    pending = launch(model, enc)
    leaves += sum(counts)
    cur = 0
    tm = {"select": 0.0, "finish": 0.0, "launch": 0.0, "step": 0.0}  # where the time goes (per shard)
    while not stop:
        other = 1 - cur
        a = time.time()
        enc, counts = groups[other].select_flat(args.leaves, args.threads)  # CPU, while the GPU works
        leaves += sum(counts)
        b = time.time()
        queued = launch(model, enc)  # queued behind the current batch: the GPU never waits for us
        c = time.time()
        finish(groups[cur], pending, args.threads)
        pending = queued
        d = time.time()
        out["replace"] = {}
        step(groups[cur], states[cur], args, rng, out)
        e = time.time()
        tm["select"] += b - a; tm["launch"] += c - b; tm["finish"] += d - c; tm["step"] += e - d
        for g, st in out["replace"].items():
            states[cur][g] = st
        for rec in out["games"]:
            log.write(json.dumps({**rec, "net_step": net_step}) + "\n")
        log.flush()
        out["games"] = []
        cur = other

        if len(out["samples"]) >= args.shard_samples:
            path = os.path.join(args.out, f"sp_{shard:06d}.npz")
            write_shard(path, out["samples"], net_step)
            dt = time.time() - t0
            print(json.dumps({"shard": shard, "samples": len(out["samples"]), "net_step": net_step,
                              "leaves_per_s": round(leaves / dt), "samples_per_s": round(len(out["samples"]) / dt, 1),
                              "time": {k: round(v / dt, 2) for k, v in tm.items()}}), flush=True)
            tm = dict.fromkeys(tm, 0.0)
            shard += 1
            written += 1
            out["samples"] = []
            t0, leaves = time.time(), 0
            if os.path.getmtime(args.net) != net_mtime:
                net_step, net_mtime = reload()
                print(f"reloaded net step {net_step}", flush=True)
            if args.max_shards and written >= args.max_shards:
                break
    if out["samples"]:
        write_shard(os.path.join(args.out, f"sp_{shard:06d}.npz"), out["samples"], net_step)
        print(f"stopped: wrote {len(out['samples'])} finished samples to shard {shard}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
