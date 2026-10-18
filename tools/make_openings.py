# Builds the frozen opening suite (PLAN §5): 8 plies picked by Apeiron at depth 7 with eval noise,
# kept only if a 1 s search scores the position within ±60 cp, deduplicated. Tooling.
#   python3 tools/make_openings.py --apeiron ~/engines/apeiron-v7.10.0/sprt --n 500 --out data/openings_classical_v1.jsonl
import argparse
import hashlib
import json
import os
import queue
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
from match import ask, full_icn, start_engine, start_referee, stop, to_token  # noqa: E402

# Classical start inside the site's engine-game border: I64_MAX - 2000 around the 1..8 start box.
BORDER = "-9223372036854773806,9223372036854773815,-9223372036854773806,9223372036854773815"
START = ('[Variant "Classical"] w 0/100 1 (8|1) ' + BORDER + " "
         "P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|"
         "R1,1+|R8,1+|r1,8+|r8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+")


NOISE = 50


def request(icn, seed, book):
    return {"icn": icn, "variant": "Classical", "wtime": 0, "btime": 0, "winc": 0, "binc": 0,
            "fixed_time": 600_000 if book else 1000, "max_depth": 7 if book else None,
            "noise_amp": NOISE if book else None, "seed": seed, "strength": None}


def candidate(seed, plies, referee, eng):
    """Plays one noisy book line. Returns (moves, score) or None if the line ended the game."""
    gid = f"c{seed}"
    ask(referee, {"cmd": "new", "id": gid, "icn": START}, 60)
    proc = start_engine(eng)
    moves = []
    try:
        for ply in range(plies):
            resp = ask(proc, request(full_icn(START, moves), seed * 100 + ply, book=True), 60)
            tok = to_token(resp["bestmove"], ply % 2 == 0)
            r = ask(referee, {"cmd": "move", "id": gid, "move": tok}, 60)
            if not r["legal"] or r["result"] is not None:
                return None
            moves.append(tok)
        stop(proc)
        proc = start_engine(eng)  # a fresh engine, so the book search's TT can't bias the score
        resp = ask(proc, request(full_icn(START, moves), seed, book=False), 60)
        return moves, resp["score"]
    finally:
        stop(proc)
        ask(referee, {"cmd": "drop", "id": gid}, 10)


def worker(seeds, out, args, eng):
    referee = start_referee()
    while True:
        try:
            seed = seeds.get_nowait()
        except queue.Empty:
            break
        out.put((seed, candidate(seed, args.plies, referee, eng)))
    stop(referee)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apeiron", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-candidates", type=int, default=900)
    ap.add_argument("--seed-start", type=int, default=0, help="first seed (dev suite: 100000, disjoint from the test suite)")
    ap.add_argument("--plies", type=int, default=8)
    ap.add_argument("--noise", type=int, default=50, help="Apeiron eval noise (cp) during book plies")
    ap.add_argument("--max-cp", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global NOISE
    NOISE = args.noise
    eng = {"bin": os.path.expanduser(args.apeiron), "threads": 1}
    seeds = queue.Queue()
    for s in range(args.seed_start, args.seed_start + args.max_candidates):
        seeds.put(s)
    out = queue.Queue()
    for _ in range(args.concurrency):
        threading.Thread(target=worker, args=(seeds, out, args, eng), daemon=True).start()

    # All candidates finish first; then the first n good ones in seed order are kept, so the
    # result does not depend on thread timing. (The 1 s scores are time-based, so a rerun can
    # still flip positions near the ±max_cp edge; the written file is what gets frozen.)
    results = {}
    for tried in range(1, args.max_candidates + 1):
        seed, res = out.get()
        results[seed] = res
        if tried % 100 == 0:
            print(f"tried {tried}/{args.max_candidates}", flush=True)

    kept, seen = [], set()
    for seed in sorted(results):
        res = results[seed]
        if res is None or abs(res[1]) > args.max_cp or tuple(res[0]) in seen:
            continue
        seen.add(tuple(res[0]))
        kept.append({"start": START, "moves": res[0], "score_cp": res[1], "seed": seed})
    tried = len(results)
    lines = [json.dumps({"id": i, **k}) for i, k in enumerate(kept[: args.n])]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"wrote {len(lines)} openings ({tried} tried, {len(kept)} passed) to {args.out}  sha256 {digest}")


if __name__ == "__main__":
    main()
