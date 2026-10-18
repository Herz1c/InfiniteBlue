# Asks Apeiron for a move in every final position the referee declared over; serve answers
# bestmove=null for positions it also sees as terminal.
import json, os, sys
sys.path.insert(0, "/mnt/c/Users/honza/Desktop/InfiniteChess/py")
from match import ask, full_icn, start_engine, stop

eng = {"bin": os.path.expanduser("~/engines/apeiron-v7.10.0/sprt"), "threads": 1}
agree = disagree = 0
for line in open(sys.argv[1]):
    g = json.loads(line)
    if g["termination"] in (None, "Maximum moves reached"):
        continue
    p = start_engine(eng)
    r = ask(p, {"icn": full_icn(g["start"], g["moves"]), "variant": "Classical", "wtime": 0, "btime": 0,
                "winc": 0, "binc": 0, "fixed_time": 100, "max_depth": None, "noise_amp": None,
                "seed": 1, "strength": None}, 30)
    stop(p)
    if r and r["bestmove"] is None:
        agree += 1
    else:
        disagree += 1
        print("Apeiron still plays", r, "after", g["termination"], g["plies"], "plies")
print(f"terminal agreement: {agree} agree, {disagree} disagree")
