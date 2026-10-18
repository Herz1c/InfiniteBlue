import sys, json, time, os, torch
sys.path.insert(0, "/mnt/c/Users/honza/Desktop/InfiniteChess/py")
import icore
from engine import load
from train import encode_batch, forward, to_device
model = load(os.path.expanduser("~/data/runs/smoke/ckpt.pt"))
icn = json.loads(open("/tmp/req.jsonl").read())["icn"]
for it in range(4):
    t0 = time.time(); pos_icn, rep = icore.position(icn); t1 = time.time()
    moves = icore.candidates(pos_icn); t2 = time.time()
    b = encode_batch([pos_icn], [rep], [""], [0.0], [0], [False], 1 << 20, 1); t3 = time.time()
    b = to_device(b, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = forward(model, b)
    torch.cuda.synchronize(); t4 = time.time()
    print(f"position {1e3*(t1-t0):.1f}ms candidates {1e3*(t2-t1):.1f}ms ({len(moves)}) encode {1e3*(t3-t2):.1f}ms forward {1e3*(t4-t3):.1f}ms")
