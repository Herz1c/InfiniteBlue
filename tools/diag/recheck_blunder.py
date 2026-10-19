# Does our search still play a move from a lost game when it runs alone (whole GPU) and longer?
#   ~/venv/ic/bin/python tools/diag/recheck_blunder.py CKPT GAMES.jsonl OPENING SIDE PLY [ms ...]
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
import engine  # noqa: E402
from match import full_icn  # noqa: E402

ckpt, games, opening, side, ply = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
budgets = [int(x) for x in sys.argv[6:]] or [5000, 20000]
g = next(g for g in map(json.loads, open(games)) if g["opening"] == opening and g["white"].startswith("m3") == (side == "w"))
icn = full_icn(g["start"], g["moves"][:ply])
model = engine.load(ckpt)
print("played in the game:", g["moves"][ply])
for ms in budgets:
    engine._tree.update(icn=None, search=None)  # fresh tree each time
    move, score, visits, _ = engine.search_move(model, icn, ms)
    print(f"{ms} ms: {move}  score {score:.0f} cp  visits {visits}", flush=True)
