# Replays Apeiron data_gen games through the site-rules referee and checks that every move is
# legal and that the game ended the way Apeiron says. Usage: python3 tools/check_games.py games.jsonl...
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# data_gen termination -> site conclusion. Anything else (adjudications, ply cap) means the
# game was stopped early, so the site must still see it as running.
APEIRON_TO_SITE = {
    "threefold repetition": "repetition",
    "fifty-move rule": "moverule",
    "insufficient_material": "insuffmat",
    "checkmate": "checkmate",
    "stalemate": "stalemate",
}


def start_referee():
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/node/bin") + ":" + os.environ["PATH"])
    cmd = [os.path.join(ROOT, "ref/node_modules/.bin/tsx"), os.path.join(ROOT, "tools/referee/referee.ts")]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)


def ask(ref, req):
    ref.stdin.write(json.dumps(req) + "\n")
    ref.stdin.flush()
    return json.loads(ref.stdout.readline())


def main(paths):
    ref = start_referee()
    counts = {}
    shown = 0
    for path in paths:
        for line in open(path):
            g = json.loads(line)
            icn = g["start_icn"] + " " + "|".join(g["moves"])
            r = ask(ref, {"cmd": "validate", "icn": icn})
            expected = APEIRON_TO_SITE.get(g["termination"])
            got = r.get("condition") if r["ok"] else "ERROR"
            agree = r["ok"] and got == expected and (expected is None or r["result"] == g["result"])
            key = (g["termination"], "agree" if agree else f"DISAGREE (site: {got})")
            counts[key] = counts.get(key, 0) + 1
            if not agree and shown < 5:
                shown += 1
                print(f"--- {path} plies={g['plies']} apeiron={g['termination']} {g['result']} site={r}")
    for (term, verdict), n in sorted(counts.items()):
        print(f"{n:5d}  {term:24s} {verdict}")
    ref.stdin.close()


if __name__ == "__main__":
    main(sys.argv[1:])
