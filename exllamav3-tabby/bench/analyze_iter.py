#!/usr/bin/env python3
"""Analyse [iter] phase-timing lines: where does a generation's time actually go?

The first instrumentation timed only the draft and verify calls, and on an agent-shaped request it
accounted for ~40% of the engine's own time_generate (14 rounds at ~65 ms against 2.27 s of
generation). This version times the whole loop iteration and splits it:

    start_ms    iterate_start_jobs - job admission and cache page work
    prefill_ms  the per-round job.prefill() loop
    recur_ms    recurrent_checkpoint()
    draft_ms    the draft model's forwards for the window
    verify_ms   the target forward, sampling and acceptance
    other_ms    the residual inside the iteration

If every phase is small and total_ms is small, but the request's wall time is large, the engine is
spending time between iterations - outside this function entirely.
"""
import re
import statistics
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/home/xujie/serve-exl3-native.log"

PAT = re.compile(
    r"\[iter\] n=(\d+) jobs=(\d+) win=(-?\d+) start_ms=([\d.]+) prefill_ms=([\d.]+) "
    r"recur_ms=([\d.]+) draft_ms=([\d.]+) verify_ms=([\d.]+) other_ms=([\d.]+) "
    r"total_ms=([\d.]+) acc=\+(-?\d+) rej=\+(-?\d+)")

rows = []
for ln in open(LOG, errors="replace"):
    m = PAT.search(ln)
    if m:
        rows.append(dict(n=int(m.group(1)), jobs=int(m.group(2)), win=int(m.group(3)),
                         start=float(m.group(4)), prefill=float(m.group(5)),
                         recur=float(m.group(6)), draft=float(m.group(7)),
                         verify=float(m.group(8)), other=float(m.group(9)),
                         total=float(m.group(10)), acc=int(m.group(11)), rej=int(m.group(12))))

if not rows:
    print("  没有 [iter] 行（服务是否用 EXL3_LOG_ROUNDS=1 启动过？）")
    raise SystemExit(0)

# Keep only the most recent contiguous run, so several tests in one log do not merge.
last_start = len(rows) - 1
while last_start > 0 and rows[last_start]["n"] - rows[last_start - 1]["n"] == 1:
    last_start -= 1
run = rows[last_start:]
print(f"  [iter] 行数 总 {len(rows)}，最近一段 {len(run)}（n={run[0]['n']}..{run[-1]['n']}）")
print()
hdr = f"  {'阶段':<12}{'最小':>8}{'中位':>8}{'最大':>8}{'合计':>10}{'占比':>7}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
tot = sum(r["total"] for r in run)
for key, label in (("start", "start_ms"), ("prefill", "prefill_ms"), ("recur", "recur_ms"),
                   ("draft", "draft_ms"), ("verify", "verify_ms"), ("other", "other_ms"),
                   ("total", "total_ms")):
    v = sorted(r[key] for r in run)
    s = sum(v)
    print(f"  {label:<12}{v[0]:>8.1f}{v[len(v) // 2]:>8.1f}{v[-1]:>8.1f}{s:>10.0f}"
          f"{100 * s / tot:>6.0f}%")

real = [r for r in run if r["win"] > 0]
print()
print(f"  有效轮（win>0）: {len(real)} / {len(run)}")
if real:
    print(f"  每轮总时中位 {statistics.median(r['total'] for r in real):.1f} ms，"
          f"每轮接受中位 {statistics.median(r['acc'] for r in real):.1f}")
    print()
    seg = max(1, len(real) // 8)
    print(f"  {'段':>4}{'win':>6}{'start':>8}{'prefill':>9}{'recur':>8}{'draft':>8}"
          f"{'verify':>8}{'other':>8}{'total':>8}{'acc':>6}")
    for i in range(0, len(real), seg):
        ch = real[i:i + seg]
        f = lambda k: sum(r[k] for r in ch) / len(ch)  # noqa: E731
        print(f"  {i // seg + 1:>4}{f('win'):>6.1f}{f('start'):>8.1f}{f('prefill'):>9.1f}"
              f"{f('recur'):>8.1f}{f('draft'):>8.1f}{f('verify'):>8.1f}{f('other'):>8.1f}"
              f"{f('total'):>8.1f}{f('acc'):>6.2f}")

    worst = sorted(real, key=lambda r: -r["total"])[:5]
    print("\n  最慢的 5 轮:")
    for r in worst:
        print(f"    n={r['n']} win={r['win']} start={r['start']:.1f} prefill={r['prefill']:.1f} "
              f"recur={r['recur']:.1f} draft={r['draft']:.1f} verify={r['verify']:.1f} "
              f"other={r['other']:.1f} total={r['total']:.1f}")

    # The headline number: does the iteration account for the wall time?
    print()
    print(f"  轮内合计 {tot / 1000:.2f} s（{len(run)} 轮）")
    print("  若请求墙钟远大于此，且各阶段都小，则时间花在 iterate() 之外")
