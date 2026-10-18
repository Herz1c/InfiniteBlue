# Search must find forced mates even with a weak net: mate-in-1 and mate-in-2 positions.
# Run in WSL: ~/venv/ic/bin/python tools/diag/search_mates.py <ckpt>
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "py"))
import icore, engine
model = engine.load(os.path.expanduser(sys.argv[1]))
CASES = [
    ("mate in 1 (two rooks + king)", "w 0/100 1 (8|1) K0,-2|R20,5|R10,1|k0,0"),
    # Black king boxed by its own pawns; the far queen drops in next to it, protected by the white king.
    ("mate in 1 (far queen, king support)", "w 0/100 1 (8|1) K2,2|Q900,1|k0,0|p-1,1|p-1,0|p-1,-1|p0,-1|p1,-1|p0,1"),
]
for name, icn in CASES:
    engine._tree["icn"] = None
    move, score, visits, _ = engine.search_move(model, icn, 2000)
    end = icore.game_end(icn + " " + move) if move else None
    print(f"{name}: {move}  visits {visits}  score {score:.0f}  -> {end}")
