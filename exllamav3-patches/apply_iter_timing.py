#!/usr/bin/env python3
"""Extend the round instrumentation to time the whole loop iteration, phase by phase.

The first version timed only the draft call and the verify call. On an agent-shaped request it
showed 14 real rounds at ~65 ms each, but the engine's own `time_generate` (time_last_token -
time_first_token) for the same request was 2.27 s - so roughly 60% of the generation happened
outside the two calls it was measuring. `recurrent_checkpoint()` and `job.prefill()` both look
cheap when there is nothing to do and neither accounts for it.

This times the full iteration instead:

    [iter] n=42 jobs=1 start_ms=0.4 prefill_ms=0.2 recur_ms=0.1 win=5 draft_ms=9.5
           verify_ms=51.3 other_ms=1.3 total_ms=62.8 acc=+2 rej=+3

`other_ms` is the residual - everything in the iteration that is not start_jobs, prefill, the
recurrent checkpoint, the draft call or the verify call. A large `other_ms` localises the gap to
loop overhead; a large `start_ms` localises it to job admission.
"""
import sys
from pathlib import Path

GEN = Path("/home/xujie/qwen38-exl3/exllamav3-spec/exllamav3/generator/generator.py")

OLD_HEAD = """        results = []
        self.iterate_start_jobs(results)

        # Perform one round of prefill
        for job in list(self.active_jobs):
            try:
                job.prefill(results)
            except Exception as e:
                self.reap_failed_job(job, e, results)

        # Recurrent checkpoints
        if self.recurrent_cache is not None:
            self.recurrent_checkpoint()
"""

NEW_HEAD = """        _it = {"t0": time.perf_counter()} if _LOG_ROUNDS else None
        results = []
        self.iterate_start_jobs(results)
        if _it is not None:
            _it["t1"] = time.perf_counter()

        # Perform one round of prefill
        for job in list(self.active_jobs):
            try:
                job.prefill(results)
            except Exception as e:
                self.reap_failed_job(job, e, results)
        if _it is not None:
            _it["t2"] = time.perf_counter()

        # Recurrent checkpoints
        if self.recurrent_cache is not None:
            self.recurrent_checkpoint()
        if _it is not None:
            _it["t3"] = time.perf_counter()
"""

OLD_LOG = """        if _rd is not None:
            _rd["t2"] = time.perf_counter()
            _rq_acc = sum(getattr(j, "accepted_draft_tokens", 0) for j in self.active_jobs)
            _rq_rej = sum(getattr(j, "rejected_draft_tokens", 0) for j in self.active_jobs)
            _rq_win = 0
            try:
                if draft_tokens is not None:
                    _rq_win = int(draft_tokens.shape[1])
            except Exception:
                _rq_win = -1
            _ROUND_N[0] += 1
            print(
                f"[round] n={_ROUND_N[0]} jobs={len(self.active_jobs)} window={_rq_win} "
                f"draft_ms={1000 * (_rd.get('t1', _rd['t0']) - _rd['t0']):.1f} "
                f"verify_ms={1000 * (_rd['t2'] - _rd.get('t1', _rd['t0'])):.1f} "
                f"total_ms={1000 * (_rd['t2'] - _rd['t0']):.1f} "
                f"acc=+{_rq_acc - _rd['acc']} rej=+{_rq_rej - _rd['rej']}",
                file=sys.stderr, flush=True,
            )
"""

NEW_LOG = """        if _it is not None:
            _it["t4"] = time.perf_counter()
            _rq_acc = sum(getattr(j, "accepted_draft_tokens", 0) for j in self.active_jobs)
            _rq_rej = sum(getattr(j, "rejected_draft_tokens", 0) for j in self.active_jobs)
            _rq_win = 0
            try:
                if draft_tokens is not None:
                    _rq_win = int(draft_tokens.shape[1])
            except Exception:
                _rq_win = -1
            _ROUND_N[0] += 1
            _rq_ms = lambda a, b: 1000 * (_it.get(b, _it[a]) - _it[a])  # noqa: E731
            _rq_total = 1000 * (_it["t4"] - _it["t0"])
            _rq_known = (_rq_ms("t0", "t1") + _rq_ms("t1", "t2") + _rq_ms("t2", "t3")
                         + _rq_ms("t3", "t4"))
            print(
                f"[iter] n={_ROUND_N[0]} jobs={len(self.active_jobs)} win={_rq_win} "
                f"start_ms={_rq_ms('t0', 't1'):.1f} prefill_ms={_rq_ms('t1', 't2'):.1f} "
                f"recur_ms={_rq_ms('t2', 't3'):.1f} draft_ms={_rq_ms('t3', 't4'):.1f} "
                f"verify_ms={_rd.get('verify', 0.0):.1f} "
                f"other_ms={max(0.0, _rq_total - _rq_known - _rd.get('verify', 0.0)):.1f} "
                f"total_ms={_rq_total:.1f} acc=+{_rq_acc - _rd['acc']} rej=+{_rq_rej - _rd['rej']}",
                file=sys.stderr, flush=True,
            )
"""


def main() -> int:
    text = GEN.read_text()
    if "[iter] n=" in text:
        print("  already extended")
        return 0
    for old, new, label in ((OLD_HEAD, NEW_HEAD, "head"), (OLD_LOG, NEW_LOG, "log")):
        if old not in text:
            print(f"FAIL  {label} anchor not found")
            return 1
        text = text.replace(old, new, 1)

    # The draft call now needs its own end timestamp recorded for the phase split, and the verify
    # call needs to be timed separately from the draft.
    old_draft = """                if _rd is not None:
                    _rd["t1"] = time.perf_counter()
                self.iterate_gen(results, draft_tokens)"""
    new_draft = """                if _it is not None:
                    _it["t3b"] = time.perf_counter()
                self.iterate_gen(results, draft_tokens)
                if _rd is not None:
                    _rd["verify"] = 1000 * (time.perf_counter() - _it["t3b"])"""
    if text.count(old_draft) != 3:
        print(f"FAIL  draft/verify anchor found {text.count(old_draft)} times, expected 3")
        return 1
    text = text.replace(old_draft, new_draft)

    # With the verify timed inside, the draft phase in the log becomes t3b - t3.
    text = text.replace(
        "f\"recur_ms={_rq_ms('t2', 't3'):.1f} draft_ms={_rq_ms('t3', 't4'):.1f} \"",
        "f\"recur_ms={_rq_ms('t2', 't3'):.1f} draft_ms={_rq_ms('t3', 't3b'):.1f} \"", 1)
    text = text.replace("_rq_total = 1000 * (_it[\"t4\"] - _it[\"t0\"])",
                        "_rq_total = 1000 * (_it.get(\"t4\", time.perf_counter()) - _it[\"t0\"])", 1)
    text = text.replace("+ _rq_ms(\"t3\", \"t4\"))",
                        "+ _rq_ms(\"t3\", \"t3b\"))", 1)
    GEN.write_text(text)
    print("  generator.py extended to per-iteration phase timing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
