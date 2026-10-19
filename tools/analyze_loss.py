# Finds where a lost game was lost: full-strength Apeiron scores the position before and after
# each of our moves, and the biggest drops (from our side's point of view) are listed.
#   ~/venv/ic/bin/python tools/analyze_loss.py GAMES.jsonl --opening 414 --as w [--ms 500] [--top 8]
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
from match import ask, full_icn, start_engine, stop  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("games")
ap.add_argument("--opening", type=int, required=True)
ap.add_argument("--as", dest="side", choices="wb", required=True, help="our colour in that game")
ap.add_argument("--ms", type=int, default=500)
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--apeiron", default=os.path.expanduser("~/engines/apeiron-v7.10.0/sprt"))
args = ap.parse_args()

game = next(g for g in map(json.loads, open(args.games))
            if g["opening"] == args.opening and g["white"].startswith("m3") == (args.side == "w"))
start, moves = game["start"], game["moves"]
eng = start_engine({"bin": args.apeiron, "threads": 1})


def score(k):
    """Apeiron's score (cp, side to move) of the position after the first k moves."""
    r = ask(eng, {"icn": full_icn(start, moves[:k]), "variant": "Classical", "wtime": 0, "btime": 0,
                  "winc": 0, "binc": 0, "fixed_time": args.ms, "max_depth": None, "noise_amp": None,
                  "seed": 1, "strength": None}, 60)
    return round(r["score"]), r["bestmove"]


ours = 0 if args.side == "w" else 1
rows = []
for k in range(len(moves)):
    if k % 2 != ours:
        continue
    before, best = score(k)  # our point of view
    after = -score(k + 1)[0] if k + 1 <= len(moves) else before  # opponent to move, so negate
    rows.append((before - after, k, moves[k], best.replace(" ", ">"), before, after))
    print(f"ply {k:3d} {moves[k]:>24}  before {before:7d}  after {after:7d}  apeiron best {best}", flush=True)
stop(eng)
print(f"\n{game['result']} {game['termination']} after {game['plies']} plies; biggest drops for us:")
for drop, k, mv, best, before, after in sorted(rows, reverse=True)[: args.top]:
    print(f"  ply {k:3d}: played {mv}, Apeiron preferred {best}; {before} -> {after} cp (drop {drop})")
