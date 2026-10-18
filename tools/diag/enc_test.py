import json, time, numpy as np, icore, os
games = [json.loads(l) for _, l in zip(range(40), open(os.path.expanduser("~/data/apeiron/classical_d12.jsonl")))]
icns, reps, moves = [], [], []
for g in games:
    for (icn, rep), mv in zip(icore.replay(g["start_icn"], g["moves"]), g["moves"]):
        icns.append(icn); reps.append(rep); moves.append(mv)
print("positions", len(icns))
for threads in (1, 4, 16):
    n = 1024
    t = time.time(); out = icore.encode(icns[:n], reps[:n], moves[:n], [False] * n, 384, threads); dt = time.time() - t
    print(f"threads {threads}: {n/dt:.0f} positions/s")
arr = {k: np.frombuffer(b, dtype=np.int32 if k in ("n_cands", "target") else np.uint8).reshape(s) for k, (b, s) in out.items()}
for k, v in arr.items(): print(k, v.shape)
tg = arr["target"]; print("target found:", (tg >= 0).mean(), "mean candidates", arr["n_cands"].mean(), "max", arr["n_cands"].max())
print("token 1 of pos 0:", arr["tokens"][0, 1], " global:", arr["tokens"][0, 0])
m = icore.encode(icns[:4], reps[:4], moves[:4], [True] * 4, 384, 1)
print("mirror changes pairs:", (np.frombuffer(m["pairs"][0], np.uint8) != arr["pairs"][:4].ravel()).any())
