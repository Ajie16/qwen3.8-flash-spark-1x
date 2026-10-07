# EXL3_SPEC_SAMPLING is a regression on this workload — measured, do not enable

**Verdict: enabling `EXL3_SPEC_SAMPLING` makes decode slower on all six content types tested, in
both window configurations. The baseline (feature off) wins everywhere. It is left disabled.**

This is the feature that was supposed to fix the acceptance problem. Its own design note
(`docs/spec_sampling.md` on the branch) projects prose acceptance rising from 45 % to 65-80 %, on
the theory that match-verify accepts only when the target samples the same token the drafter chose
(`E_p[p(d_greedy)]`, which collapses when `p` is flat) while exact speculative sampling accepts on
the total-variation overlap `E[Σ min(p, q)]`, which is provably ≥ the match rate.

Measured, it does the opposite.

## The A/B

Same six prompts, 700 tokens, service idle, paired same-session. `EXL3_MTP_HEAD_N=163840`,
`EXL3_GR_TUNED=1`, `EXL3_DRAFT_CONFIDENCE=0.6` in all three arms.

| Content | **baseline (off)** | SPEC, window 1 | SPEC, window 5 |
|---|---:|---:|---:|
| code | **58.1** | 49.1 | 39.3 |
| prose | **48.0** | 45.4 | 34.2 |
| math | **62.6** | 57.4 | 55.3 |
| json | **81.7** | 61.9 | 62.6 |
| zh_tech | **48.3** | 46.4 | 33.0 |
| zh_prose | **46.6** | 44.6 | 33.9 |

Acceptance, which is what the feature was meant to raise:

| Content | baseline | SPEC w=1 | SPEC w=5 |
|---|---:|---:|---:|
| code | 65.4 % | **81.3 %** | 32.4 % |
| prose | 54.6 % | 47.6 % | 24.7 % |
| math | 67.3 % | 73.2 % | 53.3 % |
| json | 85.5 % | 79.9 % | 62.8 % |
| zh_tech | 56.3 % | 59.0 % | 23.3 % |
| zh_prose | 49.9 % | 48.9 % | 22.8 % |

At the shipped default `EXL3_SPEC_WINDOW=1` the feature does raise per-position acceptance (code
65.4 → 81.3), but it **forces the draft window to 1**:

```python
_SPEC_WINDOW = max(1, int(_os.environ.get("EXL3_SPEC_WINDOW", "1")))
...
if _SPEC_SAMPLING:
    window = min(window, _SPEC_WINDOW)
```

so every round can yield at most 2 tokens, capping throughput near 55 tok/s. Median window fell
from 2-5 in the baseline to 1 across all six prompts. Raising it to 5 restores the window but
**halves the acceptance** (code 65.4 → 32.4, zh_tech 56.3 → 23.3) and lengthens the round from
41 ms to 66 ms, because each round now runs `log_softmax` over `EXL3_MTP_HEAD_N` elements five
times. Both costs land at once: code 58.1 → 39.3, a 32 % loss.

## Why it is plausible that this is why it was never enabled

The branch's last commit is `983f1aa` (2026-09-28, "docs: review round 2 — M2 live debugging") and
the switch has shipped with default `"0"` and was never set by any launcher. The design doc still
reads `Status: design (M0)` and its only performance figures are the *baseline* measurements plus
an "Expected" projection. It contains no result for the feature itself.

That is consistent with it having been tried and abandoned without the negative result being
written down — which is exactly the failure mode the other documents in this repo keep warning
about, and why this file exists.

## What this does not rule out

- **`EXL3_SPEC_Q_TEMP`** (default 0.6, matching the deployment's forced target temperature) was not
  swept. If the sampled draft distribution is simply a worse proposer than argmax on peaked
  targets, matching temperatures more carefully is the only plausible repair, and it was not tried
  here.
- The interaction with `EXL3_HYBRID_NGRAM` was not tested; the two are mutually exclusive by design
  (`generator.py:347-349` suppresses spec when hybrid is on).
- Only this model and this preset (`temperature 0.6 / top_p 0.95 / top_k 20 / rep_penalty 1.05`,
  which is inside the supported subset) were tested.

## Consequence for the decode-rate question

The diagnosis stands: decode rate is `(acceptance × window + 1) / round_time`, and acceptance is
the driver. The prediction formula matched the SPEC-window-5 arm to within 1 % on all six prompts
(r = +1.00), which independently confirms the formula on a third, structurally different
configuration.

But the one feature built to raise acceptance **lowers** it here. So the measured honest position
is: acceptance is the bottleneck, no available switch improves it, and the remaining options are
`EXL3_DRAFT_CONFIDENCE` tuning (the upstream sweep already found 0.6 optimal) or accepting the
current rates.

## Reproducing

```bash
# baseline (better)
bash exllamav3-tabby/serve-local.sh
python exllamav3-tabby/bench/acceptance_vs_speed.py 700

# feature on, shipped default
EXL3_SPEC_SAMPLING=1 bash exllamav3-tabby/serve-local.sh
python exllamav3-tabby/bench/acceptance_vs_speed.py 700

# feature on, window opened
EXL3_SPEC_SAMPLING=1 EXL3_SPEC_WINDOW=5 bash exllamav3-tabby/serve-local.sh
python exllamav3-tabby/bench/acceptance_vs_speed.py 700
```
