# For each root-rule miss: the position's pieces near the move's line.
import json, os, subprocess
ICORE = os.path.expanduser("~/build/icore/release/icore")
lines = []
for game in open("data/m1/games_full.icn"):
    game = game.strip(); moves = game.split()[-1].split("|")
    pos = subprocess.run([ICORE, "positions"], input=game + "\n", capture_output=True, text=True).stdout.splitlines()
    lines += [(p, m) for p, m in zip(pos, moves)]
out = subprocess.run([ICORE, "cover"], input="\n".join(f"{p}\t{m}" for p, m in lines) + "\n", capture_output=True, text=True).stdout.splitlines()
shown = 0
for (p, m), o in zip(lines, out):
    if json.loads(o)["root"]["missed"] and shown < 6:
        shown += 1
        fx, fy, tx, ty = map(int, m.split("=")[0].replace(">", ",").split(","))
        pcs = p.split()[-1].split("|")
        print(m, "| pieces:", " ".join(pcs))
