import sys, time, os, torch
sys.path.insert(0, "/mnt/c/Users/honza/Desktop/InfiniteChess/py")
from train import *
path = os.path.expanduser("~/data/shards/d12")
s = load_shard(os.path.join(path, "shard_00000.npz"))
idx = list(range(512))
t = time.time(); b = encode_batch(*take(s, idx), [False]*512, 384, 16); print(f"encode 512: {time.time()-t:.2f}s, C={b['cands'].shape[1]}")
model = Net(dec_layers=int(os.environ.get("DEC","2"))).cuda(); opt = torch.optim.AdamW(model.parameters(), 1e-3)
b = to_device(b, "cuda")
for i in range(13):
    if i == 3: torch.cuda.synchronize(); t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = forward(model, b)
    loss, _ = losses(out, b["target"].long(), b["wdl"], b["score_target"])
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize(); dt = (time.time()-t)/10
print(f"GPU train step (batch 512): {dt*1000:.0f} ms -> {512/dt:.0f} pos/s")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
    for i in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward(model, b)
        loss, _ = losses(out, b["target"].long(), b["wdl"], b["score_target"])
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
rows = sorted([e for e in prof.key_averages(group_by_input_shape=True) if e.key.startswith("aten::")], key=lambda e: -e.device_time_total)
for e in rows[:16]: print(f"{e.device_time_total/3000:7.1f} ms/step  {e.key:32s} {str(e.input_shapes)[:90]}")
