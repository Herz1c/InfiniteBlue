# Self-play starting positions (PROGRESS E12, amendment A5): the first 12 plies of Apeiron data_gen
# games, which data_gen picks at random from a MultiPV list, played from our own Classical start
# (the engine-game border, not data_gen's 1e15 border). Distinct, legal, game not over. These are
# training-corpus openings, separate from the dev and test suites.
#   ~/venv/ic/bin/python tools/make_sp_starts.py ~/data/apeiron/classical_d12.jsonl ~/data/sp_starts.txt [n_games]
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
import icore  # noqa: E402
from selfplay import start_icn  # noqa: E402

PLIES = 12
src, dst = sys.argv[1], sys.argv[2]
n_games = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
start = start_icn()
seen, out, bad = set(), [], 0
with open(src) as f:
    for i, line in enumerate(f):
        if i >= n_games:
            break
        g = json.loads(line)
        if len(g["moves"]) <= PLIES:
            continue
        icn = start + " " + "|".join(g["moves"][:PLIES])
        try:
            pos, _ = icore.position(icn)
        except ValueError:
            bad += 1
            continue
        if pos in seen or icore.game_end(icn) is not None:
            continue
        seen.add(pos)
        out.append(icn)
tmp = dst + ".tmp"
with open(tmp, "w") as f:
    f.write("\n".join(out) + "\n")
os.replace(tmp, dst)
print(f"{len(out)} distinct starts from {n_games} games ({bad} rejected as illegal under our border)")
