#!/usr/bin/env python3
"""Check the identity that must hold for every request: accepted_draft_tokens <= new_tokens.

A speculative round produces (accepted + 1) tokens: the accepted drafts plus the bonus token the
target samples after the last accepted position. So over a whole request the accepted draft count
can never exceed the output token count, whatever the draft window, acceptance rate or requeue
history.

This uses only the engine's own counters, as TabbyAPI logs them - no instrumentation, no per-round
attribution, nothing that concurrency can distort. A row where accepted > output is a request that
accepted drafts the output does not contain, i.e. work the engine did and then discarded.
"""
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/home/xujie/serve-exl3-native.log"

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

REQ = re.compile(r"#(\d+) chat/completions.*?([\d,]+) tokens generated at ([\d.]+) T/s"
                 r".*?draft (\d+)/(\d+) accepted")
PROMPT = re.compile(r"#(\d+) chat/completions.*?([\d,]+)\s+prompt tokens")
CACHED = re.compile(r"(\d+)% cached")

prompts = {}
for b in blocks:
    m = PROMPT.search(b)
    m2 = re.search(r"max_tokens:\s*(\d+)", b)
    if m and "tokens generated" not in b:
        prompts[m.group(1)] = (int(m.group(2).replace(",", "")),
                               int(m2.group(1)) if m2 else 0)

rows = []
for b in blocks:
    m = REQ.search(b)
    if not m:
        continue
    num = m.group(1)
    out = int(m.group(2).replace(",", ""))
    tps = float(m.group(3))
    acc, tot = int(m.group(4)), int(m.group(5))
    c = CACHED.search(b)
    p = prompts.get(num, (0, 0))
    rows.append((int(num), p[0], p[1], int(c.group(1)) if c else 0, out, tps, acc, tot))

print("  {:<5}{:>9}{:>8}{:>6}{:>9}{:>8}{:>10}{:>8}{:>8}  {}".format(
    "#", "prompt", "max_tok", "cach", "输出", "T/s", "接受", "总草稿", "接受≤输出", "判定"))
print("  " + "-" * 96)
bad = 0
for num, pt, mt, ch, out, tps, acc, tot in rows:
    ok = acc <= out
    if not ok:
        bad += 1
    print("  {:<5}{:>9,}{:>8,}{:>5}%{:>9,}{:>8.1f}{:>10,}{:>8,}{:>8}  {}".format(
        num, pt, mt, ch, out, tps, acc, tot, "是" if ok else "否",
        "" if ok else "★ 丢弃 {:,} tok".format(acc - out)))

print()
print("  共 {} 个请求，其中 {} 个违反恒等式".format(len(rows), bad))
if bad:
    wasted = sum(r[6] - r[4] for r in rows if r[6] > r[4])
    print("  违反者合计多接受 {:,} tok（这些是做过又被丢弃的工作）".format(wasted))
    print()
    print("  对比：")
    okr = [r for r in rows if r[6] <= r[4]]
    badr = [r for r in rows if r[6] > r[4]]
    for label, sel in (("正常", okr), ("违反", badr)):
        if sel:
            print("    {}: {} 个请求, decode 中位 {:.1f} T/s, 范围 {:.1f}-{:.1f}".format(
                label, len(sel), sorted(x[5] for x in sel)[len(sel) // 2],
                min(x[5] for x in sel), max(x[5] for x in sel)))
