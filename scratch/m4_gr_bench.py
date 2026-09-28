import time, torch
from exllamav3.ext import exllamav3_ext as ext

dev = "cuda"
H, D, LR = 4, 2560, 320
M = LR + H
torch.manual_seed(0)

streams = torch.randn(1, H, D, dtype=torch.float, device=dev)
fn_q = torch.randint(-128, 127, (M, H*D), dtype=torch.int8, device=dev)
fn_s = torch.rand(M, dtype=torch.float, device=dev)
up_q = torch.randint(-128, 127, (H, D//4, LR, 4), dtype=torch.int8, device=dev)
up_s = torch.rand(H, D, dtype=torch.float, device=dev)
w = torch.randn(H*D, dtype=torch.half, device=dev)
dots = torch.empty(1, M+1, H, dtype=torch.float, device=dev)
post = torch.empty(1, H, dtype=torch.float, device=dev)
mixed = torch.empty(1, D, dtype=torch.half, device=dev)

def timeit(fn, iters=2000, label=""):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    us = (time.perf_counter()-t0)/iters*1e6
    print(f"{label:40s} {us:7.2f} us")
    return us

bytes_pair = fn_q.numel() + up_q.numel() + streams.numel()*4*2  # tables + streams r/w approx
t = timeit(lambda: ext.gr_mix_int8(streams, fn_q, fn_s, up_q, up_s, w, 1e-6, dots, post, mixed), label="gr_mix_int8 pair R=1 (one site)")
print(f"  -> per-site {t:.1f}us, tables {bytes_pair/1e6:.2f}MB, eff BW {bytes_pair/t/1e3:.0f} GB/s; x97 sites = {97*t/1000:.2f} ms/step")

# pure-read ceiling: read both tables with a trivial reduction
def read_both():
    return fn_q.float().sum() + up_q.float().sum()  # not optimal but shows convert+reduce
t2 = timeit(lambda: torch.sum(fn_q, dtype=torch.int64) + torch.sum(up_q, dtype=torch.int64), label="pure int8 read (sum) both tables")
print(f"  -> read-only eff BW {(fn_q.numel()+up_q.numel())/t2/1e3:.0f} GB/s")

# 97 sites back-to-back to mimic per-step amortization (same tables, repeated)
t3 = timeit(lambda: [ext.gr_mix_int8(streams, fn_q, fn_s, up_q, up_s, w, 1e-6, dots, post, mixed) for _ in range(97)], iters=50, label="97 sites back-to-back (ms)")
print(f"  -> {t3/1000:.2f} ms/step equivalent")
