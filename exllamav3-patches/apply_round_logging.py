#!/usr/bin/env python3
"""Per-round instrumentation for the exllamav3 draft/verify loop.

Why: the service log's lone long requests span 14.8 to 60.3 tok/s with no server-side explanation -
concurrency, GPU throttling, context depth, memory pressure, output_chunking/requeue, constrained
generation and client read speed were each measured and ruled out, and nine synthetic workloads all
land at 45-54 tok/s. The per-request averages TabbyAPI reports cannot say *where* a round's time
went, so the next step has to be inside the loop.

What it logs, once per draft+verify round, to stderr, only when EXL3_LOG_ROUNDS=1:

    [round] n=1234 jobs=1 window=5 draft_ms=4.2 verify_ms=31.8 total_ms=36.1 acc=+3 rej=+2

- `window`    the number of draft tokens this round attempted (dynamic drafting truncates it)
- `draft_ms`  the draft model's forwards for the window
- `verify_ms` the target forward plus sampling and acceptance
- `acc`/`rej` deltas of the job-level draft counters, which is where a collapsing accept rate shows

Reading it: rounds multiplying with a healthy verify_ms means acceptance fell; verify_ms growing
with a steady window means the forward itself slowed; total_ms far above draft+verify means the
engine is waiting on something outside the loop.

Implementation notes. The flag is read once at import, like the module's other switches. The timing
points wrap the existing calls without reordering anything, so behaviour is unchanged when the flag
is off. `time` is already imported at module scope.
"""
import re
import sys
from pathlib import Path

GEN = Path("/home/xujie/qwen38-exl3/exllamav3-spec/exllamav3/generator/generator.py")

FLAG = "_LOG_ROUNDS"

FLAG_ANCHOR = "import os as _os\n"

FLAG_CODE = """import os as _os

# EXL3_LOG_ROUNDS=1 prints one line per draft+verify round to stderr. Off by default; read once at
# import like the other switches here. Used to localise a slow round when a request's average tok/s
# cannot say whether acceptance fell, the verify forward slowed, or the engine stalled outside the
# loop.
_LOG_ROUNDS = _os.environ.get("EXL3_LOG_ROUNDS", "0") != "0"
_ROUND_N = [0]
"""

OLD_BLOCK = """        # Generation with draft model
        if self.draft_model:
            if self.dflash_draft:
                draft_tokens = self.iterate_draftmodel_dflash_gen(results)
                self.iterate_gen(results, draft_tokens)
            elif self.mtp_draft:
                if self.hybrid_ngram:
                    draft_tokens = self.iterate_hybrid_gen(results)
                else:
                    draft_tokens = self.iterate_draftmodel_mtp_gen(results)
                self.iterate_gen(results, draft_tokens)
            else:
                draft_tokens = self.iterate_draftmodel_gen(results)
                self.iterate_gen(results, draft_tokens)
"""

NEW_BLOCK = """        # Generation with draft model
        _rd = None
        if _LOG_ROUNDS:
            _rd = {
                "t0": time.perf_counter(),
                "acc": sum(getattr(j, "accepted_draft_tokens", 0) for j in self.active_jobs),
                "rej": sum(getattr(j, "rejected_draft_tokens", 0) for j in self.active_jobs),
            }
        if self.draft_model:
            if self.dflash_draft:
                draft_tokens = self.iterate_draftmodel_dflash_gen(results)
                if _rd is not None:
                    _rd["t1"] = time.perf_counter()
                self.iterate_gen(results, draft_tokens)
            elif self.mtp_draft:
                if self.hybrid_ngram:
                    draft_tokens = self.iterate_hybrid_gen(results)
                else:
                    draft_tokens = self.iterate_draftmodel_mtp_gen(results)
                if _rd is not None:
                    _rd["t1"] = time.perf_counter()
                self.iterate_gen(results, draft_tokens)
            else:
                draft_tokens = self.iterate_draftmodel_gen(results)
                if _rd is not None:
                    _rd["t1"] = time.perf_counter()
                self.iterate_gen(results, draft_tokens)
        if _rd is not None:
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


def main() -> int:
    text = GEN.read_text()
    if FLAG in text and "print(\n" in text and "[round] n=" in text:
        print("  already instrumented")
        return 0
    if FLAG_ANCHOR not in text:
        print("FAIL  import anchor not found")
        return 1
    text = text.replace(FLAG_ANCHOR, FLAG_CODE, 1)
    if OLD_BLOCK not in text:
        print("FAIL  generation block not found verbatim")
        return 1
    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    if "import sys" not in text.split("\n\n")[0] and "\nimport sys" not in text[:2000]:
        text = text.replace("import os as _os\n", "import os as _os\nimport sys\n", 1)
    GEN.write_text(text)
    print("  generator.py instrumented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
