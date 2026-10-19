# How does Games.select_flat scale with threads? Splits it into the Rust call and the Python-side
# conversion of the returned buffers.
#   ~/venv/ic/bin/python tools/diag/select_scaling.py
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
import icore  # noqa: E402
import numpy as np  # noqa: E402
from selfplay import load_starts  # noqa: E402

starts = load_starts(os.path.expanduser("~/data/sp_starts.txt"))[:128]
for threads in (1, 4, 12, 16):
    g = icore.Games(starts)
    t_sel, n = 0.0, 0
    for r in range(30):
        a = time.time()
        enc, counts = g.select_flat(8, threads)
        t_sel += time.time() - a if r >= 5 else 0
        n += sum(counts) if r >= 5 else 0
        k = sum(counts); w = max(1, max(np.frombuffer(enc["n_cands"][0], dtype=np.int32), default=1))
        g.backup(np.full((k, w), 1.0 / w, np.float32).tobytes(), w, np.zeros(k, np.float32).tobytes(), threads)
    print(f"threads {threads}: select {n / t_sel:.0f} leaves/s (batch {n / 25:.0f}, width {w})", flush=True)
