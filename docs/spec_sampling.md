# Exact Speculative Sampling for MTP Drafts — Design

Status: design (M0). Target: upstream PR to vcruz305/exllamav3.

## 1. Motivation and current state

Current drafting for Qwen3.8-Flash-Next (Qwen4Exp MTP):

- **Draft** (`architecture/qwen4_exp_mtp.py:169` `sample_from_state`): greedy argmax over
  the pruned head (`EXL3_MTP_HEAD_N=110592`, GEMM output `y` is the full draft logit row).
- **Verify** (`generator/generator.py` serial loop, ~L1185): sample target at each window
  position with the job's full sampler; accept draft iff `sampled == draft`; on mismatch emit
  the sampled token and cut the window. The batch-verify fast path only engages for pure
  greedy stacks (`CustomSampler.supports_batch_verify`), so any temp/top_p/top_k/penalty
  preset takes the serial path.

Match-verify is **distribution-exact** — the emitted token's marginal is exactly p at every
position: P(emit d) = P(sample == d) = p(d); P(emit y≠d) = P(sample == y) = p(y). The
problem is purely efficiency: acceptance = E_p[p(d_greedy)], which collapses when p is flat.

Measured on the live server (Qwen3.8-Flash-Next, abliterated 3bpw, forced preset
temp 0.6 / top_p 0.95 / top_k 20 / rep_penalty 1.05, 5 draft tokens, ngram RAM on):

| workload | match acceptance | decode |
|---|---|---|
| code + thinking | ~70% | ~50-60 tok/s |
| CJK prose | ~45% | ~45 tok/s |

Exact speculative sampling (Leviathan et al. 2023) raises per-position acceptance to the
total-variation overlap E[Σ_x min(p(x), q(x))] — strictly ≥ the match rate — while keeping
the output distribution exactly p. Expected on this setup: prose 45% → 65-80% acceptance,
decode 45 → 60-70 tok/s. This is the largest remaining single-node lever.

## 2. Algorithm

Per verification window (n = num_draft_tokens):

1. **Draft**: for i in 1..n, draw d_i ~ q_i instead of argmax. q_i is the draft head's
   distribution with an optional temperature knob (`EXL3_SPEC_Q_TEMP`, default 0.6 to match
   the deployment's forced target temperature; overlap is maximized when q ≈ p in shape).
   Store per position: d_i, q_i(d_i), and the q_i row log-probs (pruned head slice,
   110592 × fp16 ≈ 216 KB/position; full window × C4 batch ≈ 4.4 MB on device — fine).
2. **Verify** (serial per position, same loop skeleton as today):
   - p_i = the job sampler's post-transform distribution over the full vocab at position i
     (temp/top_k/top_p/rep-penalty — supported subset, see §4). Past-ids–dependent steps use
     the accepted prefix so far, exactly as the serial loop already advances them.
   - Draw u ~ U[0,1) from job.rng. Accept d_i iff u < min(1, p_i(d_i)/q_i(d_i)).
   - On reject: emit t ~ renorm((p_i − q_i)⁺) and cut the window. q outside the pruned
     head slice is 0 by construction (such tokens are never drafted), so the residual there
     equals p — correct without materializing q beyond the slice.
   - On full window accept: bonus token from p at the last logit row, as today.
3. **Exactness**: standard speculative-sampling proof; the output distribution is exactly
   the job sampler's p, independent of q. q quality affects only speed.

## 3. Shadow mode (M1 — de-risk before touching the active path)

`EXL3_SPEC_SHADOW=1`: legacy match-verify executes unchanged; additionally, at each verify
position, compute and log Σ_x min(p_i(x), q_i^T(x)) for T ∈ {0.6, 0.8, 1.0} (q temperature
rescaling from the stored logit row; 3 extra softmaxes over 110k — negligible). Report
expected acceptance per workload from real traffic.

This produces the go/no-go number for M2 on real data in ~1 day, with zero behavior change.

## 4. Scope and fallbacks (legacy match-verify retained)

Opt-in: `EXL3_SPEC_SAMPLING=1` (default off until M3 validation lands).

Spec path engages only when ALL hold, else legacy path per window:

- drafter is MTP with pruned-head probs available (draft payload carries q rows)
- job sampler stack ∈ supported subset: temp, top_k, top_p, rep_penalty (+ no-op
  normalizations), multinomial draw. Anything else (DRY, min_p, XT C, banned tokens,
  logit bias, filters, forced ids, return_probs/top_tokens, CFG/multi-sequence,
  device logit mask) → legacy
- ngram draft rounds (no q) → legacy, as today

A `CustomSampler.supports_spec_sampling` flag (analogous to `supports_batch_verify`) is
computed at construction from the executed stack.

## 5. Code changes (~350-450 lines)

| file | change |
|---|---|
| `architecture/qwen4_exp_mtp.py` | new probs-export mode next to `sample_from_state`: reuse the pruned-head GEMM `y`, per-job temperature scale, fp32 log_softmax, multinomial sample from job-visible seed, return (ids, q(d), logq rows). `sample_from_state` untouched — legacy path unaffected. Other `*_mtp.py` archs can adopt later. |
| `generator/generator.py` | draft loop (`iterate_draftmodel_mtp_gen`, ~L715): carry per-job payload {ids, q_d, logq_rows} alongside `draft_ids_pinned`, incl. the device-resident (`EXL3_MTP_DEVICE_DRAFT`) chain. Verify loop (~L1185): spec branch before legacy; per-position p transform, accept test, (p−q)⁺ resample, rejection bookkeeping reusing `reject_remainder`. Shadow-mode hooks behind env flag. |
| `generator/sampler/custom.py` | `supports_spec_sampling` flag + `transformed_logprobs(logits_row, past_ids) -> (logp, sample_fn)` for the supported subset, reusing existing step math (no duplicated transform logic). |
| `generator/spec_sampling.py` (new) | pure, unit-testable core: accept test, residual (p−q)⁺ renorm + draw, RNG plumbing. No generator state. |
| `tests/` (CPU, follows dec24f8 pattern) | synthetic p/q distribution test: N windows, chi-square of emitted tokens vs p; acceptance ≈ Σ min(p,q) within tolerance; residual-draw correctness when p⊥q. |

RNG: all draws from job.rng in a fixed serial order (draft sample → per-position accept u →
residual draw), keeping runs reproducible per seed.

## 6. Validation (M3)

- **Distribution**: seed-fixed long generations spec on vs off — unigram KL within noise,
  teacher-forced mean logprob parity; the CPU chi-square test as the hard gate.
- **Quality**: needle suite 3/3 at 75k; long-output tail inspection (the 3bpw乱码 regression
  class) under the forced preset.
- **Speed** (`exl3bench.py` standard suite, hot state): targets prose ≥ 55 (from ~45),
  code ≥ 65 (from ~50-60), T5c long-ctx decode up proportionally; step overhead ≤ 1.5 ms
  (6 × full-vocab softmax + rare residual draws).

## 7. Dev infra and PR logistics

- Worktree `~/qwen38-exl3/exllamav3-spec` on branch `feat/spec-sampling` (from master
  74b6f5a); live service keeps running the master checkout untouched.
- A/B: second TabbyAPI on :8898 with PYTHONPATH pointed at the worktree; same model dir.
- origin is vcruz305/exllamav3 (no push access): at M3 either fork under the user's GitHub
  and open the PR from there, or send the patch series on an issue. Open a short tracking
  issue with this design + shadow-mode numbers before the PR.
