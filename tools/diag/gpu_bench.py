# Measure inference throughput (positions/s) of a piece-token transformer on this GPU.
import time, torch, torch.nn as nn

dev = "cuda"
print(torch.cuda.get_device_name(0), torch.__version__)

def bench(layers, d, heads, tokens, batch, dtype=torch.bfloat16, iters=30):
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True, norm_first=True),
        layers).to(dev, dtype).eval()
    x = torch.randn(batch, tokens, d, device=dev, dtype=dtype)
    bias = torch.randn(batch * heads, tokens, tokens, device=dev, dtype=dtype)  # relational bias as attn mask
    params = sum(p.numel() for p in enc.parameters())
    with torch.inference_mode():
        for _ in range(5):
            enc(x, mask=bias)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            enc(x, mask=bias)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / iters
    print(f"L={layers:2d} d={d:4d} h={heads:2d} tok={tokens} B={batch:5d} params={params/1e6:6.2f}M  "
          f"{batch/dt:9.0f} pos/s  {dt*1e3:6.2f} ms/batch")

# matmul peak
a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
for _ in range(3): a @ a
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): a @ a
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 20
print(f"bf16 matmul: {2*8192**3/dt/1e12:.1f} TFLOPS")

for layers, d, heads in [(6, 128, 4), (8, 256, 8), (10, 384, 8), (12, 512, 8)]:
    for batch in (256, 2048):
        bench(layers, d, heads, 40, batch)
