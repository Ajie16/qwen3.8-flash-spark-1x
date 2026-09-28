#!/usr/bin/env python3
"""M4: decompose the ~34ms/step EXL3 decode floor on GB10.

Loads the 3bpw pack (no draft model), prefills a ~N-token prompt, then runs
pure greedy decode batch-1: warmup steps, then profiled steps under
torch.profiler (CPU+CUDA). Per-step wall clock is measured with
cuda.synchronize around each gen.iterate().

Outputs per ctx:
  <prefix>.ctx<N>.trace.json   chrome trace
  <prefix>.ctx<N>.keyavg.txt   key_averages sorted by self device time
  <prefix>.ctx<N>.wall.json    per-step wall times + prefill stats
"""
import argparse, json, os, sys, threading, time
import torch

FILLER = ("The maintenance log for reactor bay seven records a pressure excursion at "
          "oh four hundred hours, followed by a manual override and a return to nominal. "
          "Subsequent inspection found no fault in the primary loop. ")

def build(ntok, salt):
    reps = max(1, int(ntok * 4 / len(FILLER)))
    return "Record %d. " % salt + FILLER * reps + "\n\nSummarize the above in one sentence."

def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return 0.0

def watchdog(floor_gib):
    while True:
        m = mem_avail_gib()
        if m < floor_gib:
            print(f"WATCHDOG: MemAvailable {m:.2f} GiB < {floor_gib} GiB, aborting", flush=True)
            os._exit(3)
        time.sleep(1.0)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ctxs", default="500,8000")
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--floor-gib", type=float, default=2.0)
ap.add_argument("--out-prefix", required=True)
a = ap.parse_args()

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
from torch.profiler import profile, ProfilerActivity

threading.Thread(target=watchdog, args=(a.floor_gib,), daemon=True).start()
print(f"exllamav3 at {__import__('exllamav3').__file__}", flush=True)
print(f"MemAvailable before load: {mem_avail_gib():.1f} GiB", flush=True)

t0 = time.time()
config = Config.from_directory(a.model)
model = Model.from_config(config)
maxctx = max(int(c) for c in a.ctxs.split(","))
maxctx = max(4096, ((maxctx + 512 + 255) // 256) * 256)
cache = Cache(model, max_num_tokens=maxctx, max_history=0, max_batch_size=1)
model.load(progressbar=False)
tokenizer = Tokenizer.from_config(config)
print(f"model loaded in {time.time()-t0:.0f}s, MemAvailable {mem_avail_gib():.1f} GiB", flush=True)

gen = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=2048)

def drain():
    for r in gen.iterate():
        pass

for ctx in [int(c) for c in a.ctxs.split(",")]:
    total_new = a.warmup + a.steps
    ids = tokenizer.encode(build(ctx, 42), add_bos=False)
    ptok = int(ids.shape[-1])
    job = Job(input_ids=ids, max_new_tokens=total_new, min_new_tokens=total_new,
              sampler=ArgmaxSampler(), seed=0)
    gen.enqueue(job)

    # prefill (ends when the first decode token exists)
    tp0 = time.perf_counter()
    while job.new_tokens < 1 and gen.num_remaining_jobs():
        drain()
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - tp0

    # warmup decode steps (first token already counts as step 1)
    for _ in range(a.warmup - 1):
        drain()
    torch.cuda.synchronize()

    # profiled decode steps
    walls = []
    tall0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(a.steps):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            drain()
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t1)
    tall = time.perf_counter() - tall0

    prefix = f"{a.out_prefix}.ctx{ctx}"
    prof.export_chrome_trace(prefix + ".trace.json")
    try:
        tbl = prof.key_averages().table(sort_by="self_device_time_total", row_limit=40)
    except Exception:
        tbl = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
    with open(prefix + ".keyavg.txt", "w") as f:
        f.write(tbl)
    walls_ms = [w * 1000 for w in walls]
    sw = sorted(walls_ms)
    meta = {
        "ctx_target": ctx, "prompt_tokens": ptok,
        "prefill_s": round(prefill_s, 3),
        "prefill_tok_s": round(ptok / prefill_s, 1),
        "steps": a.steps,
        "wall_mean_ms": round(sum(walls_ms) / len(walls_ms), 3),
        "wall_p50_ms": round(sw[len(sw) // 2], 3),
        "wall_min_ms": round(sw[0], 3),
        "wall_max_ms": round(sw[-1], 3),
        "wall_total_s_prof_region": round(tall, 3),
        "walls_ms": [round(w, 3) for w in walls_ms],
    }
    with open(prefix + ".wall.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"CTX {ctx}: ptok={ptok} prefill={prefill_s:.2f}s "
          f"wall mean={meta['wall_mean_ms']}ms p50={meta['wall_p50_ms']}ms "
          f"min={meta['wall_min_ms']} max={meta['wall_max_ms']}", flush=True)

print("DONE", flush=True)
