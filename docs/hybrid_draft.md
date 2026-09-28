# M3: Hybrid ngram + MTP drafting

`EXL3_HYBRID_NGRAM=1` (default off = master behavior). Every round, each job drafts
from its suffix-automaton continuation when the automaton finds a repeat of at least
`EXL3_NGRAM_MIN_MATCH` tokens (default 8) with continuation left to copy — near-free on
repetitive content (code, markup, templates) where repeats accept in long runs — and
falls back to the MTP draft otherwise. Both sources share the legacy match-verify in
`iterate_gen`, so outputs keep baseline (non-spec) sampling semantics.

The pre-existing pure-ngram mode (`ngram_match_min`, mutually exclusive with a draft
model) is untouched.

## Knobs

| env | default | meaning |
|---|---|---|
| `EXL3_HYBRID_NGRAM` | 0 | master switch; requires the MTP drafter |
| `EXL3_NGRAM_MIN_MATCH` | 8 | minimum suffix-repeat length to engage ngram |
| `EXL3_NGRAM_MAX_DRAFT` | 16 | max continuation tokens drafted per ngram round |
| `EXL3_HYBRID_STATS` | 0 | `hybrid-stats r=.. ng=.. na=.. ma=..` every 50 rounds: ngram round share, per-position ngram acceptance, per-position MTP acceptance |
| `EXL3_NGRAM_ADAPTIVE` | 1 | per-job backoff: disengage ngram below `EXL3_NGRAM_MIN_ACC` acceptance, re-probe later |
| `EXL3_NGRAM_MIN_ACC` | 0.25 | per-position ngram acceptance floor (0.25 × 16-wide window ≈ the MTP round's yield) |
| `EXL3_NGRAM_ADAPT_ROUNDS` | 16 | ngram rounds between engagement evaluations |
| `EXL3_NGRAM_PROBE_ROUNDS` | 64 | rounds a disengaged job waits before re-probing |

## Round structure (`Generator.iterate_hybrid_gen`)

1. Probe every prefill-done job's SAM (`Job.probe_ngram_draft`). `accept_tensor`
   consumes history incrementally and rebuilds on rewind, so intermittent probing
   (ngram this round, MTP the next) is exact; a non-qualifying probe still does the
   bookkeeping that keeps the automaton current.
2. No candidates: pure MTP round, identical to the un-hybrid path.
3. All candidates: pure ngram round — the MTP draft forward is skipped entirely.
4. Mixed: the regular full-batch MTP draft runs (sub-batching the drafter would save
   ~4 ms against a ~34 ms target-forward floor but desync the calibrator and shadow
   row maps), then ngram rows overwrite their MTP rows. Rows are padded to the round's
   combined window with token 0; a pad position verifies like any draft position —
   accepts iff the target samples the pad token (exact, just unlikely), rejects the
   remainder otherwise.

## MTP state alignment (why ngram rounds are safe mid-stream)

The MTP carry (`job.mtp_last_hidden`) and the draft KV are maintained by `iterate_gen`
from the **target** forward, not by the drafter: `draft_verifier_params` are added to
the target forward whenever a draft model is attached (regardless of draft source), the
post-verify `draft_model.prefill` writes the accepted tokens — whoever proposed them —
into the draft cache, and `mtp_last_hidden` is re-paired with the last accepted
position. An ngram round therefore advances MTP state exactly like an MTP round, and
MTP drafting resumes on any later round from a consistent carry. The drafter's
speculative KV positions are position-indexed scratch rewritten every round, so a
skipped draft forward leaves nothing stale.

## Interactions

- **Spec sampling (M2)**: suppressed whenever hybrid is on (one warning at init). An
  ngram round has no draft q, and mixing per-row verify semantics mid-round leaves
  neither the M2-exact nor the legacy distribution. Shadow observation
  (`EXL3_SPEC_SHADOW`) still works on pure-MTP rounds; rounds with any ngram row are
  skipped because the shadow's logq rows describe discarded MTP drafts.
- **Confidence calibrator**: pure ngram rounds produce no labels. In mixed rounds,
  ngram rows are skipped (their MTP conf/ids pairs were never tested) and MTP rows
  clamp acceptance to the real draft width (pad accepts have no conf entry).
- **Page budget**: `draft_reserve_tokens` is bumped to `EXL3_NGRAM_MAX_DRAFT` so a wide
  ngram window never outruns the job's allocated pages.
- **Recurrent state history**: GDN conv/state buffers are sized by the frontend from
  the draft window (tabbyAPI: `max_history = draft_num_tokens`). A wider ngram window
  overflows them — the generator clamps `EXL3_NGRAM_MAX_DRAFT` to `cache.max_history`
  as a safety net, and the frontend should size `max_history` for the wider window
  (the local tabbyAPI patch does, env-gated on `EXL3_HYBRID_NGRAM`).

## Failure mode: structural repeats with varying content

A long suffix repeat does not imply a repeating continuation — JSON record lists share
long key prefixes but diverge at the values. The first live run accepted only ~8% per
position on such content (and a wide window costs more than an MTP window once the
continuation diverges), hence the per-job adaptive backoff: below
`EXL3_NGRAM_MIN_ACC` per-position acceptance the job returns to MTP drafting and
re-probes every `EXL3_NGRAM_PROBE_ROUNDS` rounds. Exact repetition (boilerplate,
tables, templates) accepts 40%+ per position and stays engaged.

CPU gates: `tests/test_hybrid_draft_cpu.py` (standalone, no pytest).
