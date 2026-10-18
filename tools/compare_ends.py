# M1 test (c): replays whole games with our rules (`icore replay`) and with the site's referee.
# Every move must be legal for us, and both must agree on whether and how each game ended.
#   python3 tools/compare_ends.py games.icn        (one whole-game ICN per line)
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))
from match import ask, start_referee  # noqa: E402

ICORE = os.path.expanduser("~/build/icore/release/icore")


def main(path):
    games = [l.strip() for l in open(path) if l.strip()]
    res = subprocess.run([ICORE, "replay"], input="\n".join(games) + "\n", capture_output=True, text=True, check=True)
    ours = [json.loads(l) for l in res.stdout.splitlines()]
    referee = start_referee()
    counts = {}
    for icn, o in zip(games, ours):
        r = ask(referee, {"cmd": "validate", "icn": icn}, 300)
        site = (r["result"], r["condition"]) if r["ok"] else ("error", r["error"][:80])
        mine = (o["end"]["result"], o["end"]["condition"]) if o["end"] else (None, None)
        if not o["ok"]:
            key = f"OURS REJECTS ply {o['ply']}"
        elif mine == site:
            key = f"agree: {site[1]}"
        else:
            key = f"DISAGREE ours={mine} site={site}"
        counts[key] = counts.get(key, 0) + 1
    for k, n in sorted(counts.items()):
        print(f"{n:5d}  {k}")


if __name__ == "__main__":
    main(sys.argv[1])
