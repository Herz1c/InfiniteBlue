# Packs Apeiron data_gen games into training shards: every position (as an ICN, replayed and
# legality-checked by our Rust rules) with its labels. 500 games per shard; a shard is written
# atomically, so stopping at any moment loses at most the shard in progress. Restarting skips the
# shards already on disk. With --follow it keeps waiting for data_gen to append more games.
#   python py/pack.py --games ~/data/apeiron/classical_d12.jsonl --out ~/data/shards/d12 [--follow]
import argparse
import json
import os
import signal
import time

import icore
import numpy as np

GAMES_PER_SHARD = 500
OPENING_PLIES = 12  # data_gen picks these moves at random from a MultiPV list: no policy target
RESULT = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

stop = False


def on_signal(*_):
    global stop
    stop = True


def blob(strings):
    """Variable-length strings as one byte array plus offsets."""
    data = [s.encode() for s in strings]
    off = np.zeros(len(data) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(d) for d in data])
    return np.frombuffer(b"".join(data), dtype=np.uint8), off


def pack_games(lines, first_game):
    icns, reps, moves, score, stm, result, ply, game = [], [], [], [], [], [], [], []
    skipped = 0
    for gi, line in enumerate(lines):
        g = json.loads(line)
        try:
            positions = icore.replay(g["start_icn"], g["moves"])
        except ValueError:
            skipped += 1  # never seen so far; kept as a safety net
            continue
        labels = {p["ply"]: p for p in g["positions"]}
        for p, ((icn, rep), mv) in enumerate(zip(positions, g["moves"])):
            lab = labels.get(p)
            if lab is None:
                continue
            icns.append(icn)
            reps.append(rep)
            moves.append(mv if p >= OPENING_PLIES else "")
            score.append(lab["score"])
            stm.append(0 if lab["stm"] == "w" else 1)
            result.append(RESULT[g["result"]])
            ply.append(p)
            game.append(first_game + gi)
    icn_blob, icn_off = blob(icns)
    move_blob, move_off = blob(moves)
    return dict(
        icn_blob=icn_blob, icn_off=icn_off, move_blob=move_blob, move_off=move_off,
        rep=np.array(reps, np.uint8), score=np.array(score, np.int32), stm=np.array(stm, np.uint8),
        result=np.array(result, np.int8), ply=np.array(ply, np.int16), game=np.array(game, np.int32),
    ), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--follow", action="store_true", help="wait for more games instead of exiting")
    args = ap.parse_args()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    os.makedirs(args.out, exist_ok=True)

    done = len([f for f in os.listdir(args.out) if f.endswith(".npz")])
    f = open(args.games)
    for _ in range(done * GAMES_PER_SHARD):
        f.readline()
    print(f"resuming after {done} shards", flush=True)
    pending = []
    while not stop:
        line = f.readline()
        if line.endswith("\n"):
            pending.append(line)
        else:
            if line:
                f.seek(f.tell() - len(line))  # a game still being written: read it again later
            if not args.follow:
                break
            time.sleep(30)
            continue
        if len(pending) == GAMES_PER_SHARD:
            shard, skipped = pack_games(pending, done * GAMES_PER_SHARD)
            path = os.path.join(args.out, f"shard_{done:05d}.npz")
            np.savez(path + ".tmp.npz", **shard)
            os.replace(path + ".tmp.npz", path)
            print(f"{path}: {len(shard['rep'])} positions, {skipped} games skipped", flush=True)
            done += 1
            pending = []
    print("stopped", flush=True)


if __name__ == "__main__":
    main()
