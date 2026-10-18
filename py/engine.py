# Our engine, speaking Apeiron's `serve` protocol (one JSON request per line on stdin, one JSON
# reply per line on stdout), so py/match.py can play it against Apeiron.
#   --mode policy  (M2) plays the candidate with the highest policy logit, no search
#   --mode search  (M3) MCTS for the request's fixed_time: Rust owns the tree (icore.Search),
#                  this loop evaluates batches of leaves on the GPU
#   python py/engine.py --ckpt ~/data/runs/m2a/ckpt.pt --mode search
import argparse
import json
import sys
import time

import icore
import numpy as np
import torch

from net import Net
from train import K, encode_batch, forward, to_device


def load(path):
    ck = torch.load(path, map_location="cuda")
    a = ck["args"]
    model = Net(a["d"], a["layers"], a["heads"]).cuda().eval()
    model.load_state_dict(ck["model"])
    return model


@torch.no_grad()
def best_move(model, icn):
    """(move token, score in cp for the side to move), or (None, 0) without legal moves."""
    pos_icn, rep = icore.position(icn)
    moves = icore.candidates(pos_icn)
    if not moves:
        return None, 0.0
    # No cap on candidates, so the encoder's order equals icore.candidates' order.
    b = encode_batch([pos_icn], [rep], [""], [0.0], [0], [False], 1 << 20, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _, score = forward(model, to_device(b, "cuda"))
    return moves[int(logits[0, : len(moves)].argmax())], float(score[0]) * K


def _launch(model, enc):
    """Starts the net on an encoded batch without waiting for the GPU (CUDA runs asynchronously)."""
    b = to_device({k: torch.from_numpy(np.frombuffer(v, dtype=np.int32 if k in ("n_cands", "target") else np.uint8).reshape(sh).copy())
                   for k, (v, sh) in enc.items()}, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _, score = forward(model, b)
    return torch.softmax(logits, dim=1), 2 * torch.sigmoid(score) - 1, b["n_cands"]


def _finish(s, pending):
    """Waits for a launched batch and backs it up into the tree."""
    priors, values, counts = pending
    counts = counts.tolist()
    s.backup([p[:c] for p, c in zip(priors.cpu().tolist(), counts)], values.cpu().tolist())


# Tree kept between moves of one game (an engine process plays one game in match.py).
_tree = {"icn": None, "search": None}


def _split(icn):
    """(position part, move tokens) of an ICN whose last token may be a move list."""
    head, _, tail = icn.rpartition(" ")
    return (head, tail.split("|")) if ">" in tail else (icn, [])


@torch.no_grad()
def search_move(model, icn, ms, batch=64, threads=8, contempt=0.0, reuse=True):
    """MCTS for `ms` milliseconds. Returns (move token, score in cp, root visits, reused visits).
    The CPU selects the next batch while the GPU evaluates the current one; the tree of the
    previous move is reused when the new position continues the same game."""
    deadline = time.monotonic() + 0.9 * ms / 1000
    s, reused = None, 0
    prev = _tree["icn"]
    if reuse and prev is not None:
        (ph, pm), (nh, nm) = _split(prev), _split(icn)
        if ph == nh and nm[: len(pm)] == pm and _tree["search"].advance(nm[len(pm):]):
            s = _tree["search"]
            reused = s.visits()
    if s is None:
        s = icore.Search(icn)
    s.set_params(2.0, 0.3, contempt)
    enc, n = s.select(batch, threads)
    pending = _launch(model, enc) if n else None
    if not n:
        s.backup([], [])
    while True:
        enc, n = s.select(batch, threads)  # overlaps with the GPU working on `pending`
        if pending is not None:
            _finish(s, pending)
        pending = _launch(model, enc) if n else None
        if not n:
            s.backup([], [])
        if time.monotonic() > deadline and (s.visits() > 0 or time.monotonic() > deadline + 1.0):
            break
    if pending is not None:
        _finish(s, pending)
    # The root stays at this position; the next request's extra moves (ours + the reply) re-root it.
    _tree["icn"], _tree["search"] = icn, s
    best = s.best()
    if best is None:
        return None, 0.0, 0, reused
    move, _, q = best
    q = max(-0.999, min(0.999, q))
    return move, K * float(np.log((1 + q) / (1 - q))), s.visits(), reused


def apeiron_format(token):
    """Our "x,y>x,y=Q" -> the serve protocol's "x,y x,y q"."""
    squares, _, promo = token.partition("=")
    return squares.replace(">", " ") + (" " + promo.lower() if promo else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=("policy", "search"), default="policy")
    ap.add_argument("--contempt", type=float, default=0.0, help="draw value -contempt for us (search mode)")
    ap.add_argument("cmd", nargs="?", default="serve")  # accepted for compatibility with Apeiron
    args = ap.parse_args()
    model = load(args.ckpt)
    # Warm up CUDA before the first request; otherwise the first move costs about a second.
    best_move(model, "w 0/100 1 (8|1) K5,1|Q4,1|R1,1|k5,8|q4,8|r1,8")
    for line in sys.stdin:
        line = line.strip()
        if line == "quit":
            break
        if not line:
            continue
        t = time.monotonic()
        try:
            req = json.loads(line)
            if args.mode == "search":
                move, score, visits, reused = search_move(model, req["icn"], req.get("fixed_time") or 1000, contempt=args.contempt)
            else:
                (move, score), visits = best_move(model, req["icn"]), 0
            reply = {"bestmove": apeiron_format(move) if move else None, "score": score, "visits": visits, "panic": None}
        except Exception as e:  # reported to the match runner as an engine failure
            reply = {"bestmove": None, "score": None, "panic": repr(e)}
        reply["elapsed_ms"] = round((time.monotonic() - t) * 1000)
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
