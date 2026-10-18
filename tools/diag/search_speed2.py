# Search speed with pipelining, and tree reuse across a short self-continued game.
# Run in WSL: ~/venv/ic/bin/python tools/diag/search_speed2.py <ckpt>
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "py"))
import engine
model = engine.load(os.path.expanduser(sys.argv[1]))
engine.best_move(model, "w 0/100 1 (8|1) K5,1|Q4,1|R1,1|k5,8|q4,8|r1,8")
ops = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "..", "..", "bench", "openings_dev_v1.jsonl"))][:3]
for o in ops:
    icn = o["start"] + " " + "|".join(o["moves"])
    engine._tree["icn"] = None
    rates = []
    for ply in range(6):  # the engine plays both sides; each move should reuse the subtree
        t = time.time()
        move, sc, visits, reused = engine.search_move(model, icn, 2000)
        dt = time.time() - t
        if move is None:
            print(f"ply {ply}: game over")
            break
        rates.append((visits - reused) / dt)
        print(f"ply {ply}: {move:28s} visits {visits:6d} reused {reused:6d}  new/s {(visits - reused) / dt:6.0f}")
        icn = icn + "|" + move
    print(f"mean new visits/s {sum(rates) / len(rates):.0f}")
