#!/usr/bin/env python3
"""Fix the requeue token accounting, and log each requeue so it is countable.

The bug, in prepare_for_requeue():

    "rq_new_tokens": self.new_tokens,   # every token accepted so far counts; ...

The comment says "every token accepted so far counts", but `self.new_tokens` is only the CURRENT
segment - `__init__` resets it to 0 for the requeued job. The result is added back at report time:

    "new_tokens": self.rq_new_tokens + self.new_tokens,

so on the second and later requeues the running total `self.rq_new_tokens` is overwritten by the
current segment's count and every earlier segment is lost from the reported output.

`accepted_draft_tokens` is carried differently - `self.accepted_draft_tokens` already includes all
prior segments, and the constructor reads it straight back - so it does NOT lose them. That
asymmetry is exactly the observable defect: `accepted_draft_tokens > new_tokens`, which is
impossible for a speculative job since a round emits accepted + 1 tokens.

Measured on this service: 1 of 14 requests violated the identity, and it was the slow one -
20.1 tok/s against a 51.8 median - with 10,581 accepted against 4,111 reported output.

This applies the one-line fix and adds a `[requeue]` line per requeue so the count per request is
visible in the log rather than inferred.
"""
import sys
from pathlib import Path

JOB = Path("/home/xujie/qwen38-exl3/exllamav3-spec/exllamav3/generator/job.py")

OLD = '            "rq_new_tokens": self.new_tokens,   # every token accepted so far counts; the requeued segment starts after them\n'

NEW = ('            # `self.rq_new_tokens` already holds every earlier segment; `self.new_tokens` is only\n'
       '            # this one. Adding them here is what keeps the reported total correct across the 2nd and\n'
       '            # later requeues - assigning self.new_tokens alone silently drops the earlier segments, and\n'
       '            # since accepted_draft_tokens is carried whole, the two counters then disagree.\n'
       '            "rq_new_tokens": self.rq_new_tokens + self.new_tokens,\n')

MARK = "self.rq_new_tokens + self.new_tokens,"

# Requeue event log, placed where the generator re-enqueues.
GEN = Path("/home/xujie/qwen38-exl3/exllamav3-spec/exllamav3/generator/generator.py")

RQ_OLD = "            rq_job = job.prepare_for_requeue()\n"

RQ_NEW = ('            if _LOG_ROUNDS:\n'
          '                print(\n'
          '                    f"[requeue] ser={getattr(job, \'serial_number\', -1)} "\n'
          '                    f"new_tokens={job.new_tokens} rq_new_tokens={job.rq_new_tokens} "\n'
          '                    f"accepted={job.accepted_draft_tokens} kv={job.sequences[0].kv_position}",\n'
          '                    file=sys.stderr, flush=True,\n'
          '                )\n'
          '            rq_job = job.prepare_for_requeue()\n')


def main() -> int:
    t = JOB.read_text()
    if "self.rq_new_tokens + self.new_tokens,\n" in t and '"rq_new_tokens"' not in t.replace(MARK, ""):
        print("  job.py already fixed")
    else:
        if OLD not in t:
            print("FAIL  rq_new_tokens assignment not found verbatim")
            return 1
        JOB.write_text(t.replace(OLD, NEW, 1))
        print("  job.py: rq_new_tokens now accumulates")

    g = GEN.read_text()
    if "[requeue]" in g:
        print("  generator.py already logs requeues")
    else:
        if RQ_OLD not in g:
            print("FAIL  requeue anchor not found")
            return 1
        GEN.write_text(g.replace(RQ_OLD, RQ_NEW, 1))
        print("  generator.py: [requeue] lines added")
    return 0


if __name__ == "__main__":
    sys.exit(main())
