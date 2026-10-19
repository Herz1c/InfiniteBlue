# Memory check for tree reuse (E7b incident, 2026-10-19): one engine plays both sides of a long
# game with search_move (repetitions are avoided by random moves) and prints its resident memory every 20 plies. With the fix in
# mcts.rs::advance (compaction) memory should level off instead of growing with the game.
#   ~/venv/ic/bin/python tools/diag/mem_test.py CKPT [plies] [ms]
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
import icore  # noqa: E402
from engine import load, search_move  # noqa: E402
from match import full_icn  # noqa: E402


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) // 1024
    return -1


ckpt = sys.argv[1]
plies = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ms = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
root = os.path.join(os.path.dirname(__file__), "..", "..")
opening = json.loads(open(os.path.join(root, "bench", "openings_dev_v1.jsonl")).readline())
model = load(ckpt)
moves = list(opening["moves"])
for ply in range(plies):
    icn = full_icn(opening["start"], moves)
    if icore.game_end(icn):
        print("game over at ply", ply, icore.game_end(icn))
        break
    move, score, visits, reused = search_move(model, icn, ms)
    # Keep the game long: a move that repeats a position is replaced by a random candidate.
    if icore.position(full_icn(opening["start"], moves + [move]))[1] > 0:
        move = random.choice(icore.candidates(icn))
    moves.append(move)
    if ply % 20 == 0:
        print(f"ply {ply}: rss {rss_mb()} MB, visits {visits}, reused {reused}", flush=True)
print(f"end: rss {rss_mb()} MB")
