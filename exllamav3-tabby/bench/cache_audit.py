#!/usr/bin/env python3
"""Prefix-cache hit rate per request, and what the misses cost.

Every request line reports the prompt size, how much of it was cached, and how many tokens had to be
prefilled. If requests that share a long prefix report `none cached`, the cache is not matching and
every turn re-prefills the whole context - which shows up as tens of seconds of time-to-first-token
while the prefill kernel itself runs at full speed.
"""
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/home/xujie/serve-normal.log"

blocks, cur = [], ""
for ln in open(LOG, errors="replace"):
    ln = ln.rstrip("\n")
    if re.match(r"^\d\d:\d\d:\d\d\.\d+ ", ln) or ln.startswith("==>") or ln.startswith("warning:"):
        if cur:
            blocks.append(cur)
        cur = ln
    else:
        cur += " " + ln.strip()
if cur:
    blocks.append(cur)

# "prompt 19,338 tokens, none cached, 19,338 new in 20.6 s (939 T/s)"
# "prompt 14,586 tokens, 81% cached, 2,810 new in 9.08 s (309 T/s)"
CACHED = re.compile(r"prompt ([\d,]+) tokens, (\d+)% cached, ([\d,]+) new in ([\d.]+) s \(([\d,]+) T/s\)")
COLD = re.compile(r"prompt ([\d,]+) tokens, none cached, ([\d,]+) new in ([\d.]+) s \(([\d,]+) T/s\)")
DONE = re.compile(r"#(\d+) chat/completions")
FIRST = re.compile(r"first token ([\d.]+) s")
OUT = re.compile(r"([\d,]+) tokens generated at ([\d.]+) T/s")

rows = []
for b in blocks:
    d = DONE.search(b)
    if not d or "tokens generated at" not in b:
        continue
    num = int(d.group(1))
    m = CACHED.search(b)
    if m:
        pt = int(m.group(1).replace(",", ""))
        ch, new = int(m.group(2)), int(m.group(3).replace(",", ""))
        pf, ttft = float(m.group(4)), None
    else:
        m = COLD.search(b)
        if not m:
            continue
        pt, ch = int(m.group(1).replace(",", "")), 0
        new, pf = int(m.group(2).replace(",", "")), float(m.group(3))
    f = FIRST.search(b)
    o = OUT.search(b)
    rows.append((num, pt, ch, new, pf, float(f.group(1)) if f else None,
                 int(o.group(1).replace(",", "")) if o else None))

print(f"  解析到 {len(rows)} 个请求\n")
zero = [r for r in rows if r[2] == 0]
print(f"  0% 命中（全冷）: {len(zero)} 个")
print(f"  有命中:          {len(rows) - len(zero)} 个")
if rows:
    hits = [r[2] for r in rows]
    print(f"  命中率中位:      {sorted(hits)[len(hits)//2]}%")
print()
print("  {:<5}{:>9}{:>8}{:>9}{:>9}{:>10}".format("#", "prompt", "缓存%", "新算", "prefill s", "首token s"))
print("  " + "-" * 52)
for num, pt, ch, new, pf, ttft, out in rows[-28:]:
    print("  {:<5}{:>9,}{:>8}{:>9,}{:>9.2f}{:>10}".format(
        num, pt, ch, new, pf, "{:.2f}".format(ttft) if ttft else "-"))

print()
cold_cost = sum(r[4] for r in rows if r[2] == 0)
warm_cost = sum(r[4] for r in rows if r[2] > 0)
print(f"  全冷请求的 prefill 总耗时: {cold_cost:.1f} s")
print(f"  有命中请求的 prefill 总耗时: {warm_cost:.1f} s")
