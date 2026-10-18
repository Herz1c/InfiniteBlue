# M1 test (d): does our candidate list contain the moves Apeiron plays, and Apeiron's root lists?
#   python3 tools/coverage.py data/m1/games_full.icn ref/perft/positions.icn
import json
import os
import subprocess
import sys

ICORE = os.path.expanduser("~/build/icore/release/icore")
APEIRON_PERFT = os.path.expanduser("~/src/apeiron-v7.10.0/target/release/examples/perft_icn")


def run(cmd, lines):
    out = subprocess.run(cmd, input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def summarize(title, results):
    for rule in ("tree", "root"):
        t = {k: sum(r[rule][k] for r in results) for k in ("exact", "far", "gap", "missed", "illegal")}
        n = sum(t.values())
        cands = sum(r[rule]["n_candidates"] for r in results) / max(1, len(results))
        print(f"{title} [{rule}] moves={n}  exact={t['exact']} ({100 * t['exact'] / max(1, n):.2f}%)  far={t['far']}  gap={t['gap']}  missed={t['missed']} "
              f"({100 * t['missed'] / max(1, n):.3f}%)  illegal={t['illegal']}  avg candidates/position={cands:.0f}")
    for rule in ("tree", "root"):
        missed = [m for r in results for m in r[rule]["missed_moves"]]
        if missed:
            print(f"   first missed ({rule}):", missed[:12])


def main(games_path, probes_path):
    # Played moves: position before each move, paired with that move.
    lines = []
    for game in open(games_path):
        game = game.strip()
        if not game:
            continue
        moves = game.split()[-1].split("|")
        positions = run([ICORE, "positions"], [game])
        lines += [f"{p}\t{m}" for p, m in zip(positions, moves)]
    summarize("played moves", [json.loads(l) for l in run([ICORE, "cover"], lines)])

    # Apeiron's own root candidate lists on the 24 probes.
    probes = [l.strip() for l in open(probes_path) if l.strip() and not l.startswith("#")]
    roots = [l.split(" ", 2)[2].split(" ") for l in run([APEIRON_PERFT, "1", "dump"], probes) if l.startswith("ROOT")]
    lines = [f"{icn}\t{'|'.join(r)}" for icn, r in zip(probes, roots)]
    summarize("apeiron root lists", [json.loads(l) for l in run([ICORE, "cover"], lines)])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
