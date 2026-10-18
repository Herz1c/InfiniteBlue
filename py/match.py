# Engine-vs-engine match runner. Both engines speak Apeiron's `serve` protocol (one JSON request
# per line on stdin, one JSON reply per line on stdout); our engine will speak it too. Every move
# goes through the site-rules referee, which also decides how the game ends. Tooling.
#
#   python3 py/match.py --a ~/engines/apeiron-v7.10.0/sprt --a-strength 8 \
#                       --b ~/engines/apeiron-v7.10.0/sprt --b-strength 1 \
#                       --openings data/openings_classical_v1.jsonl --pairs 20 --ms 200 --out games.jsonl
import argparse
import json
import os
import queue
import select
import shlex
import signal
import subprocess
import threading
import time

import sprt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_STRENGTH = 8  # Apeiron's full strength; sent as null like its own harness does

TERMINATION = {
    "checkmate": "Checkmate",
    "stalemate": "Stalemate",
    "repetition": "Threefold repetition",
    "moverule": "50-move rule",
    "insuffmat": "Insufficient material",
}


SPAWNED = []  # every engine and referee process, so a stop request can kill them all


def spawn(cmd, env=None):
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, env=env,
                            start_new_session=True)  # Ctrl-C / job stop reaches only match.py
    SPAWNED.append(p)
    return p


def ask(proc, req, timeout_s):
    """One request, one reply line. None on timeout or a dead process."""
    try:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
    except BrokenPipeError:
        return None
    ready, _, _ = select.select([proc.stdout], [], [], timeout_s)
    if not ready:
        return None
    line = proc.stdout.readline()
    return json.loads(line) if line else None


def start_referee():
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/node/bin") + ":" + os.environ["PATH"])
    return spawn([os.path.join(ROOT, "ref/node_modules/.bin/tsx"),
                  os.path.join(ROOT, "tools/referee/referee.ts")], env)


def start_engine(eng):
    """`bin` is an Apeiron binary (run as "<bin> serve") or a full command line, e.g.
    "python py/engine.py --ckpt run/ckpt.pt"."""
    env = dict(os.environ, APEIRON_THREADS=str(eng["threads"]))
    cmd = shlex.split(eng["bin"])
    return spawn(cmd if len(cmd) > 1 else cmd + ["serve"], env)


def stop(proc):
    try:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def to_token(bestmove, white_to_move):
    """Apeiron's "x,y x,y[ q]" -> the site's "x,y>x,y[=Q]" (promotion letter cased by colour)."""
    parts = bestmove.split()
    tok = parts[0] + ">" + parts[1]
    if len(parts) > 2:
        tok += "=" + (parts[2].upper() if white_to_move else parts[2].lower())
    return tok


def full_icn(start, moves):
    return start + (" " + "|".join(moves) if moves else "")


def play_game(white, black, opening, ms, max_plies, referee, gid, seed):
    """Plays one game. Returns a dict with result ("1-0", "0-1", "1/2-1/2") and details."""
    start = opening["start"]
    moves = list(opening["moves"])
    r = ask(referee, {"cmd": "new", "id": gid, "icn": full_icn(start, moves)}, 60)
    if not r or not r["ok"]:
        raise RuntimeError(f"referee rejected opening {opening['id']}: {r}")
    procs = {"w": start_engine(white), "b": start_engine(black)}
    engines = {"w": white, "b": black}
    spent = {"w": [], "b": []}
    result, termination = r["result"], TERMINATION.get(r["condition"])

    while result is None:
        if len(moves) >= max_plies:
            result, termination = "1/2-1/2", "Maximum moves reached"
            break
        side = "w" if len(moves) % 2 == 0 else "b"  # every opening starts with White to move
        eng = engines[side]
        req = {"icn": full_icn(start, moves), "variant": "Classical",
               "wtime": 0, "btime": 0, "winc": 0, "binc": 0,
               "fixed_time": ms, "max_depth": None, "noise_amp": None, "seed": seed + len(moves),
               "strength": None if eng["strength"] >= MAX_STRENGTH else eng["strength"]}
        t0 = time.monotonic()
        # An engine's first request also covers its startup (e.g. loading a net onto the GPU).
        first = not spent[side]
        resp = ask(procs[side], req, timeout_s=180.0 if first else max(10.0, 20 * ms / 1000))
        wall_ms = (time.monotonic() - t0) * 1000
        loser_result = "0-1" if side == "w" else "1-0"
        if resp is None or resp.get("panic") or not resp.get("bestmove"):
            result, termination = loser_result, f"Engine failure ({resp})"
            break
        used = resp.get("elapsed_ms", wall_ms)
        spent[side].append(round(used))
        if used > 2 * ms:
            result, termination = loser_result, f"Time forfeit ({used:.0f} ms)"
            break
        tok = to_token(resp["bestmove"], side == "w")
        r = ask(referee, {"cmd": "move", "id": gid, "move": tok}, 60)
        if not r["ok"] or not r["legal"]:
            result, termination = loser_result, f"Illegal move {tok} ({r.get('reason') or r.get('error')})"
            break
        moves.append(tok)
        result, termination = r["result"], TERMINATION.get(r["condition"])

    for p in procs.values():
        stop(p)
    ask(referee, {"cmd": "drop", "id": gid}, 10)
    return {"result": result, "termination": termination, "plies": len(moves),
            "moves": moves, "ms_white": spent["w"], "ms_black": spent["b"]}


def worker(jobs, done, args, a, b):
    referee = start_referee()
    while True:
        try:
            pair, g, opening = jobs.get_nowait()
        except queue.Empty:
            break
        white, black = (a, b) if g == 0 else (b, a)
        out = play_game(white, black, opening, args.ms, args.max_plies, referee,
                        gid=f"{pair}-{g}", seed=1000 * pair + 100 * g)
        out.update(pair=pair, game=g, opening=opening["id"], white=white["name"], black=black["name"],
                   start=opening["start"])
        done.put(out)
    stop(referee)


def score_for_a(game, a_name):
    s = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[game["result"]]
    return s if game["white"] == a_name else 1.0 - s


def pentanomial(games, a_name):
    pairs = {}
    for g in games:
        pairs.setdefault(g["pair"], []).append(score_for_a(g, a_name))
    penta = [0, 0, 0, 0, 0]
    for scores in pairs.values():
        if len(scores) == 2:
            penta[int(sum(scores) * 2)] += 1
    return penta


def summary(games, a_name):
    w = sum(1 for g in games if score_for_a(g, a_name) == 1.0)
    l = sum(1 for g in games if score_for_a(g, a_name) == 0.0)
    d = len(games) - w - l
    pairs = {}
    for g in games:
        pairs.setdefault(g["pair"], []).append(score_for_a(g, a_name))
    # Pentanomial: per finished pair, A's total score 0, 0.5, 1, 1.5 or 2.
    penta = [0, 0, 0, 0, 0]
    for scores in pairs.values():
        if len(scores) == 2:
            penta[int(sum(scores) * 2)] += 1
    n = len(games)
    score = (w + d / 2) / n if n else 0.0
    return f"A: +{w} ={d} -{l}  score {score:.3f}  pentanomial[LL,LD,WL+DD,WD,WW]={penta}"


def main():
    ap = argparse.ArgumentParser()
    for side in ("a", "b"):
        ap.add_argument(f"--{side}", required=True, help="engine binary speaking the serve protocol")
        ap.add_argument(f"--{side}-strength", type=int, default=MAX_STRENGTH)
        ap.add_argument(f"--{side}-threads", type=int, default=1)
        ap.add_argument(f"--{side}-name")
    ap.add_argument("--openings", required=True, help="JSONL of {id, start, moves}")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--opening-offset", type=int, default=0, help="first opening to use (e.g. 200 for a confirmation run)")
    ap.add_argument("--ms", type=int, default=1000, help="fixed time per move")
    ap.add_argument("--max-plies", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"),
                    help="stop once the pentanomial SPRT decides (logistic Elo, alpha=beta=0.05)")
    ap.add_argument("--min-pairs", type=int, default=20)
    args = ap.parse_args()

    def engine(side):
        e = {"bin": os.path.expanduser(getattr(args, side)) if " " not in getattr(args, side) else getattr(args, side),
             "strength": getattr(args, f"{side}_strength"),
             "threads": getattr(args, f"{side}_threads")}
        e["name"] = getattr(args, f"{side}_name") or f"{side.upper()}:{os.path.basename(os.path.dirname(e['bin'].split()[0]))}:L{e['strength']}"
        return e

    a, b = engine("a"), engine("b")
    openings = [json.loads(line) for line in open(args.openings)]
    # Resumable: games already in --out are kept and not replayed.
    games = [json.loads(line) for line in open(args.out)] if os.path.exists(args.out) else []
    finished = {(g["pair"], g["game"]) for g in games}
    if games:
        print(f"resuming: {len(games)} games already in {args.out}", flush=True)
    jobs = queue.Queue()
    for pair in range(args.pairs):
        for g in (0, 1):
            if (pair, g) not in finished:
                jobs.put((pair, g, openings[(args.opening_offset + pair) % len(openings)]))
    if args.sprt and sum(pentanomial(games, a["name"])) >= args.min_pairs             and sprt.report(pentanomial(games, a["name"]), *args.sprt)[1] != "continue":
        jobs = queue.Queue()  # the SPRT already decided before this resume: play nothing more
    # A stop request (SIGINT/SIGTERM) ends the run without recording the games in progress, so an
    # interrupted game is never counted; they are replayed on resume.
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    done = queue.Queue()
    threads = [threading.Thread(target=worker, args=(jobs, done, args, a, b), daemon=True)
               for _ in range(args.concurrency)]
    for t in threads:
        t.start()

    with open(args.out, "a") as f:
        while True:
            if stop.is_set():
                try:
                    print(f"STOPPED by request; {len(games)} games kept in {args.out}", flush=True)
                except OSError:
                    pass  # stdout may be a pipe whose reader was stopped too
                for p in SPAWNED:
                    if p.poll() is None:
                        p.kill()
                os._exit(130)  # the worker threads die with the process
            try:
                g = done.get(timeout=1)
            except queue.Empty:
                if all(not t.is_alive() for t in threads):
                    break  # every game finished (or the SPRT stopped scheduling new ones)
                continue
            games.append(g)
            f.write(json.dumps(g) + "\n")
            f.flush()
            print(f"[{len(games)}/{2 * args.pairs}] pair {g['pair']} {g['white']} vs {g['black']}: "
                  f"{g['result']} {g['termination']} ({g['plies']} plies)  |  {summary(games, a['name'])}",
                  flush=True)
            if args.sprt:
                penta = pentanomial(games, a["name"])
                line, decision = sprt.report(penta, *args.sprt)
                print("   SPRT", line, flush=True)
                if decision != "continue" and sum(penta) >= args.min_pairs:
                    # Stop scheduling; games already running still finish and count.
                    while not jobs.empty():
                        try:
                            jobs.get_nowait()
                        except queue.Empty:
                            break
    print("FINAL", summary(games, a["name"]))
    if args.sprt:
        print("FINAL SPRT", sprt.report(pentanomial(games, a["name"]), *args.sprt)[0])


if __name__ == "__main__":
    main()
