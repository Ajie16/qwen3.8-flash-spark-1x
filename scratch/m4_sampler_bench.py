import os, time, torch, random
from exllamav3.generator.sampler import CustomSampler, SS_RepP, SS_Temperature, SS_TopK, SS_TopP, SS_Sample, SS_Argmax

dev = "cuda"
V = 248832          # padded vocab used by the pack head
AV = 248320         # actual vocab
torch.manual_seed(0)

def bench(sampler, logits, past, iters=200, label=""):
    # warmup
    for _ in range(10):
        sampler.forward(logits, sequence_ids=past, rand_u32=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        sampler.forward(logits, sequence_ids=past, rand_u32=1)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    # also CPU-only launch cost (no sync per call, already measured in aggregate)
    print(f"{label:44s} {dt:7.3f} ms/call")
    return dt

for rows in (1, 4):
    logits = torch.randn(rows, 1, V, dtype=torch.half, device=dev) * 5
    past = torch.randint(0, AV, (rows, 500), dtype=torch.long, device=dev)
    serve = CustomSampler([SS_RepP(1.05, int(10e7), 1), SS_Temperature(0.6), SS_TopK(20), SS_TopP(0.95), SS_Sample()])
    print("serve stack steps:", [type(s).__name__ for s in serve.steps])
    bench(serve, logits, past, label=f"serve stack (rep1.05+t0.6+k20+p0.95) rows={rows}")
    greedy = CustomSampler([SS_Argmax()])
    bench(greedy, logits, past, label=f"argmax only rows={rows}")
    fused_only = CustomSampler([SS_Temperature(0.6), SS_TopK(20), SS_TopP(0.95), SS_Sample()])
    print("fused-only steps:", [type(s).__name__ for s in fused_only.steps])
    bench(fused_only, logits, past, label=f"fused tail only rows={rows}")
    repp = CustomSampler([SS_RepP(1.05, int(10e7), 1), SS_Argmax()])
    print("repp steps:", [type(s).__name__ for s in repp.steps])
    bench(repp, logits, past, label=f"rep-penalty + argmax rows={rows}")
