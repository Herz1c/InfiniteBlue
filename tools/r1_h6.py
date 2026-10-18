# PROGRESS E9 (H6, pre-registered 2026-10-17 19:08): cost of playing our gap/far representative
# instead of Apeiron's own square, judged by Apeiron. Writes data/r1/h6_cases.jsonl and prints the
# pre-registered statistics. Run inside WSL:  ~/venv/ic/bin/python tools/r1_h6.py
import glob
import json
import os
import queue
import sys
import threading

import icore
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))
from match import ask, start_engine, stop  # noqa: E402

APEIRON = {"bin": os.path.expanduser("~/engines/apeiron-v7.10.0/sprt"), "threads": 1}
MS = 500
CLIP = 1000
OUT = os.path.join(ROOT, "data", "r1", "h6_cases.jsonl")
# E9: the 152 data_gen benchmark games. E9b: --corpus <jsonl> --games 2000 --max-cases 1500.
CORPUS = None
MAX_GAMES = None
MAX_CASES = None


def collect():
    cases = []
    paths = [CORPUS] if CORPUS else sorted(glob.glob(os.path.expanduser("~/dg/*.jsonl")))
    for path in paths:
        for gi, line in enumerate(open(path)):
            if MAX_GAMES is not None and gi >= MAX_GAMES:
                break
            if MAX_CASES is not None and len(cases) >= MAX_CASES:
                return cases
            g = json.loads(line)
            for ply, ((icn, _rep), mv) in enumerate(zip(icore.replay(g["start_icn"], g["moves"]), g["moves"])):
                c = icore.h6_case(icn, mv, len(cases) + 1)
                if c:
                    cases.append({"src": os.path.basename(path), "game": gi, "ply": ply, "icn": icn, "own": mv,
                                  "kind": c[0], "rep": c[1], "rand": c[2]})
    return cases


def score_after(proc, icn, move, seed):
    """Apeiron's score of the position after `move`, from the mover's side (cp, clipped)."""
    r = ask(proc, {"icn": f"{icn} {move}", "variant": "Classical", "wtime": 0, "btime": 0, "winc": 0, "binc": 0,
                   "fixed_time": MS, "max_depth": None, "noise_amp": None, "seed": seed, "strength": None}, 60)
    if not r or r.get("score") is None:
        return None
    return float(np.clip(-r["score"], -CLIP, CLIP))


def worker(jobs, results):
    proc = start_engine(APEIRON)
    while True:
        try:
            i, c = jobs.get_nowait()
        except queue.Empty:
            break
        c = dict(c)
        c["s_own"] = score_after(proc, c["icn"], c["own"], 1)
        c["s_own2"] = score_after(proc, c["icn"], c["own"], 2)
        c["s_rep"] = score_after(proc, c["icn"], c["rep"], 1)
        c["s_rand"] = score_after(proc, c["icn"], c["rand"], 1)
        results.put(c)
    stop(proc)


def ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x))).mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    global CORPUS, MAX_GAMES, MAX_CASES, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--games", type=int)
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    CORPUS, MAX_GAMES, MAX_CASES, OUT = a.corpus and os.path.expanduser(a.corpus), a.games, a.max_cases, a.out
    cases = collect()[:MAX_CASES] if MAX_CASES else collect()
    print(f"{len(cases)} cases ({sum(c['kind'] == 'gap' for c in cases)} gap, {sum(c['kind'] == 'far' for c in cases)} far)", flush=True)
    jobs, results = queue.Queue(), queue.Queue()
    for i, c in enumerate(cases):
        jobs.put((i, c))
    threads = [threading.Thread(target=worker, args=(jobs, results)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = [results.get() for _ in range(results.qsize())]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if None not in (r["s_own"], r["s_own2"], r["s_rep"], r["s_rand"])]
    print(f"scored {len(ok)} of {len(rows)} (others hit a finished game)")
    for kind in ("all", "gap", "far"):
        sel = [r for r in ok if kind == "all" or r["kind"] == kind]
        if not sel:
            continue
        print(f"--- {kind} (n={len(sel)})")
        for name, other in (("representative", "s_rep"), ("noise (own twice)", "s_own2"), ("random other gap", "s_rand")):
            loss = np.array([r["s_own"] - r[other] for r in sel])
            m, lo, hi = ci(loss)
            print(f"  {name:18s} mean loss {m:7.1f} cp  95% CI [{lo:6.1f}, {hi:6.1f}]  median {np.median(loss):6.1f}  "
                  f"share <= 50 cp {np.mean(loss <= 50):.3f}")
    loss = np.array([r["s_own"] - r["s_rep"] for r in ok])
    m, lo, hi = ci(loss)
    share = np.mean(loss <= 50)
    verdict = "SUPPORTED" if hi <= 20 and share >= 0.90 else "REJECTED" if lo > 50 else "NOT DEMONSTRATED"
    print(f"H6 verdict by the pre-registered rule: {verdict} (mean CI upper {hi:.1f} <= 20 and share {share:.3f} >= 0.90?)")


if __name__ == "__main__":
    main()
