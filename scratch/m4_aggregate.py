#!/usr/bin/env python3
"""Aggregate a chrome trace: GPU busy/span/gap, kernel count, launches, buckets, top kernels."""
import json, sys, re, collections

path, steps = sys.argv[1], int(sys.argv[2])
ev = json.load(open(path))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev

kernels = [e for e in ev if e.get("cat") == "kernel"]
memcpy = [e for e in ev if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
launches = [e for e in ev if e.get("cat") == "cuda_runtime" and "LaunchKernel" in e.get("name", "")]

busy = sum(e["dur"] for e in kernels) + sum(e["dur"] for e in memcpy)
gpu_ev = kernels + memcpy
span = max(e["ts"] + e["dur"] for e in gpu_ev) - min(e["ts"] for e in gpu_ev)
print(f"kernels: {len(kernels)} total, {len(kernels)/steps:.0f}/step; "
      f"memcpy/memset: {len(memcpy)} ({sum(e['dur'] for e in memcpy)/1000:.2f}ms)")
print(f"cudaLaunchKernel: {len(launches)} total, {len(launches)/steps:.0f}/step, "
      f"CPU time {sum(e['dur'] for e in launches)/1000:.1f}ms ({sum(e['dur'] for e in launches)/1000/steps:.2f}ms/step)")
print(f"GPU busy {busy/1000:.1f}ms over span {span/1000:.1f}ms -> per step busy {busy/1000/steps:.2f}ms, "
      f"idle-in-span {(span-busy)/1000/steps:.2f}ms/step, occupancy {busy/span*100:.1f}%")

BUCKETS = [
    ("MoE GEMM+routing", r"exl3_moe_coop|exl3_mgemm|routing_"),
    ("dense GEMM (attn proj/lm_head/dense)", r"exl3_gemm|gemvx|aten::mm|deinterleave|cublas|sm1\d+_xmma|nvjet"),
    ("GDN (gated delta net)", r"gated_delta|gdn_|conv1d_update|causal_conv"),
    ("attention (QSA/paged)", r"attn|_qsa_|_paged_kv|_mla_|flash|fa_"),
    ("hyperconnection/gr mixers", r"gr_finalize|gr_dots|hc_apply"),
    ("norm/elementwise/rope", r"rms|norm|rope|silu|elementwise|mul_sigmoid|sigmoid|vectorized|copy|cat_|index"),
    ("sampler", r"sample|argmax|topk|softmax|sort"),
]
agg = collections.defaultdict(lambda: [0, 0.0])
names = collections.defaultdict(lambda: [0, 0.0])
for e in kernels:
    n = e["name"]
    names[n][0] += 1; names[n][1] += e["dur"]
    for label, pat in BUCKETS:
        if re.search(pat, n, re.I):
            agg[label][0] += 1; agg[label][1] += e["dur"]; break
    else:
        agg["other"][0] += 1; agg["other"][1] += e["dur"]

print(f"\n{'bucket':42s} {'ms/step':>8s} {'%busy':>6s} {'k/step':>7s}")
for label, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"{label:42s} {d/1000/steps:8.2f} {d/busy*100:5.1f}% {c/steps:7.1f}")

print(f"\ntop 20 kernels by total time:")
print(f"{'name':70s} {'ms/step':>8s} {'%busy':>6s} {'n/step':>7s} {'us avg':>7s}")
for n, (c, d) in sorted(names.items(), key=lambda kv: -kv[1][1])[:20]:
    print(f"{n[:70]:70s} {d/1000/steps:8.2f} {d/busy*100:5.1f}% {c/steps:7.1f} {d/c:7.1f}")
