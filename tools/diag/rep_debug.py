import json, subprocess, sys, os
sys.path.insert(0, "/mnt/c/Users/honza/Desktop/InfiniteChess/py")
from match import ask, start_referee
ICORE = os.path.expanduser("~/build/icore/release/icore")
games = [l.strip() for l in open("data/m1/games_full.icn") if l.strip()]
ours = [json.loads(l) for l in subprocess.run([ICORE, "replay"], input="\n".join(games) + "\n", capture_output=True, text=True).stdout.splitlines()]
ref = start_referee()
for icn, o in zip(games, ours):
    r = ask(ref, {"cmd": "validate", "icn": icn}, 300)
    if r["condition"] == "repetition" and (o["end"] is None):
        pos = subprocess.run([ICORE, "positions"], input=icn + "\n", capture_output=True, text=True).stdout.splitlines()
        def key(p):  # drop clock/limit and fullmove
            t = p.split()
            return " ".join(x for x in t if "/" not in x and not x.isdigit())
        final = key(pos[-1])
        hits = [i for i, p in enumerate(pos) if key(p) == final]
        print("plies", len(pos) - 1, "final position occurs at", hits)
        for i in hits:
            print("  ", pos[i][:60], "...")
        moves = icn.split()[-1].split("|")
        print("  moves from first hit:", moves[hits[0]:])
        for i in range(36, len(pos)):
            toks = pos[i].split()
            pcs = sorted(toks[-1].split("|"))
            print(i, toks[1], "moved:", moves[i-1] if i else "-", "| K/R:", [p for p in pcs if p[0] in "KkRr"], "| pieces", len(pcs))
        break
