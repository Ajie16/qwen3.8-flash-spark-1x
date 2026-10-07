#!/usr/bin/env python3
"""Correlate the identity violation with the per-request parameters that could trigger a rewind.

TabbyAPI logs the resolved request settings, including `stop: N sequences (req)` and
`banned_strings`. `receive_sample`'s docstring names the three things that make it hold output and
potentially rewind to a checkpoint: token healing, a partial Unicode character or stop string, and
banned-string handling. So if the violating request is the one carrying stop sequences, the trigger
is identified rather than guessed.
"""
import re

LOG = "/home/xujie/serve-exl3-native.log"

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

params, done = {}, {}
for b in blocks:
    m = re.search(r"#(\d+) chat/completions", b)
    if not m:
        continue
    num = m.group(1)
    if "max_tokens:" in b and "tokens generated at" not in b:
        params[num] = {
            "stop": "stop:" in b,
            "banned": "banned_strings" in b,
            "heal": "token_healing" in b,
            "maxtok": (re.search(r"max_tokens:\s*(\d+)", b) or [None, "?"])[1],
            "prompt": (re.search(r"([\d,]+)\s+prompt tokens", b) or [None, "0"])[1],
        }
    m2 = re.search(r"#(\d+) chat/completions.*?([\d,]+) tokens generated at ([\d.]+) T/s"
                   r".*?draft (\d+)/(\d+) accepted", b)
    if m2:
        done[num] = (int(m2.group(2).replace(",", "")), float(m2.group(3)),
                     int(m2.group(4)), int(m2.group(5)))

print("  {:<5}{:>8}{:>7}{:>9}{:>8}{:>10}{:>9}{:>7}".format(
    "#", "prompt", "stop", "banned", "heal", "输出", "T/s", "接受"))
print("  " + "-" * 66)
viol = []
for num in sorted(done, key=lambda k: int(k)):
    out, tps, acc, tot = done[num]
    p = params.get(num, {})
    row = (num, p.get("prompt", "?"), p.get("stop"), p.get("banned"), p.get("heal"), out, tps, acc)
    mark = ""
    if acc > out:
        viol.append(row)
        mark = "  <== 违反，丢弃 {:,}".format(acc - out)
    print("  {:<5}{:>8}{:>7}{:>9}{:>8}{:>10,}{:>9.1f}{:>7,}{}".format(
        num, str(row[1]), str(row[2]), str(row[3]), str(row[4]), out, tps, acc, mark))

print()
withstop = [r for r in viol if r[2]]
print("  违反恒等式的请求: {}".format(
    ", ".join("#{} (stop={})".format(r[0], r[2]) for r in viol) if viol else "无"))
print("  其中带 stop 串的: {}/{}".format(len(withstop), len(viol)))
allstop = [r for r in [tuple(done[n]) + (params.get(n, {}).get("stop"),) for n in done] if r[-1]]
print("  全部带 stop 串的请求数: {}".format(len(allstop)))
