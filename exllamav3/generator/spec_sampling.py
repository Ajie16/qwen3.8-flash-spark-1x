from __future__ import annotations
import math
import os as _os
import torch
from dataclasses import dataclass
from .sampler.custom import (
    SS_RepP,
    SS_Fused,
    SS_Temperature,
    SS_TopK,
    SS_TopP,
    SS_Sample,
    SS_Normalize,
    SS_Sort,
)

"""
Exact speculative sampling (design: docs/spec_sampling.md) — pure, generator-state-free
helpers. M1 uses them for shadow-mode measurement (EXL3_SPEC_SHADOW=1); M2 uses them for
the real spec path (EXL3_SPEC_SAMPLING=1): drafts sampled from the shaped q, per-position
acceptance min(1, p/q) and rejection resampling from renorm((p - q)+), all in log space.

spec_transform() mirrors the executed sampler stack

    [SS_RepP?, SS_Fused(MODE_SAMPLE | MODE_SAMPLE_FILTERS)]

(or the equivalent discrete tail when EXL3_FUSED_SAMPLER=0) as a standalone
logits -> log-probs transform over the full vocabulary. Known negligible differences vs.
the fused CUDA kernel:

- top_k truncates exactly K tokens in sort order; the fused kernel keeps every token tied
  exactly at the cutoff
- rep-penalty factors reduce with torch scatter_reduce("amax") instead of the kernel's
  shared-memory atomicMax
"""

# Draft distribution shaping (M1 shadow winner on live traffic: q_shaped edged out
# q_t0.6/q_t0.8/q_t1.0). Exactness holds for any q; these only affect acceptance rate.
SPEC_Q_TEMP = float(_os.environ.get("EXL3_SPEC_Q_TEMP", "0.6"))
SPEC_Q_TOP_K = 20
SPEC_Q_TOP_P = 0.95


@dataclass
class SpecTransform:
    """
    Supported-subset sampler parameters extracted from a CustomSampler's executed stack.
    """
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    temp_first: bool = True
    rep_p: float = 1.0
    sustain_range: int = 0
    decay_range: int = 0


def extract_spec(sampler) -> SpecTransform | None:
    """
    Extract the shadow/spec transform from a CustomSampler's executed stack. Returns None
    when the stack is outside the supported subset (min_p, greedy, DRY/XTC/pres-freq
    penalties, bans, logit bias, extra steps, ...); the caller falls back to legacy-only
    behavior for that job.
    """
    steps = getattr(sampler, "steps", None)
    if not steps:
        return None
    tail = list(steps)
    spec = SpecTransform()

    # Head: at most one repetition-penalty step
    if type(tail[0]) is SS_RepP:
        rep = tail.pop(0)
        spec.rep_p = rep.rep_p
        spec.sustain_range = rep.sustain_range
        spec.decay_range = rep.decay_range
    if not tail:
        return None

    # Fused tail (default, EXL3_FUSED_SAMPLER=1)
    if len(tail) == 1 and type(tail[0]) is SS_Fused:
        f = tail[0]
        if f.mode == SS_Fused.MODE_SAMPLE:
            spec.temperature = 1.0 / f.inv_temp
            return spec
        if f.mode == SS_Fused.MODE_SAMPLE_FILTERS:
            spec.temperature = 1.0 / f.inv_temp
            spec.top_k = f.top_k
            spec.top_p = f.top_p
            # SS_Fused weighs the filter histogram with the sampling temperature iff
            # temperature preceded the filters in the original stack
            spec.temp_first = math.isclose(f.inv_temp_filter, f.inv_temp)
            return spec
        # MODE_GREEDY (degenerate, spec gain is zero) and MODE_SAMPLE_MINP (min_p is
        # outside the v1 subset) fall back to legacy
        return None

    # Discrete tail (EXL3_FUSED_SAMPLER=0, testing/validation): SS_Normalize / SS_Sort are
    # inserted prep steps that do not change the transform math
    rest = [s for s in tail if type(s) not in (SS_Normalize, SS_Sort)]
    if not rest or not isinstance(rest[-1], SS_Sample):
        return None
    pre = rest[:-1]

    # At most one temperature step, leading or trailing the filter sequence
    temp_first = False
    temperature = 1.0
    if pre and type(pre[0]) is SS_Temperature:
        temp_first = True
        temperature = pre[0].temperature
        pre = pre[1:]
    if pre and type(pre[-1]) is SS_Temperature:
        if temp_first:
            return None
        temperature = pre[-1].temperature
        pre = pre[:-1]

    # Filters must be a subsequence of (TopK, TopP); min_p and everything else -> legacy
    order = (SS_TopK, SS_TopP)
    pos = -1
    for s in pre:
        t = type(s)
        if t not in order or order.index(t) <= pos:
            return None
        pos = order.index(t)

    spec.temperature = temperature
    spec.temp_first = temp_first
    spec.top_k = next((s.top_k for s in pre if type(s) is SS_TopK), 0)
    spec.top_p = next((s.top_p for s in pre if type(s) is SS_TopP), 1.0)
    return spec


def _apply_rep_penalty(logits: torch.Tensor, past_ids: torch.Tensor, spec: SpecTransform) -> torch.Tensor:
    """
    Torch mirror of ext.apply_rep_pens: per-token penalty factor (max over occurrences,
    linearly decayed between sustain_range and sustain_range + decay_range), positive
    logits divided by rep_p, negative ones multiplied, interpolated by the factor.
    """
    past = past_ids.view(-1)
    past_len = past.numel()
    sustain = spec.sustain_range
    decay = spec.decay_range
    vocab = logits.shape[-1]
    factors = torch.zeros(vocab, dtype = torch.float32, device = logits.device)
    lo = past_len - sustain - decay
    # Kernel skips i <= past_len - sustain_range - decay_range
    if past_len and lo < past_len - 1:
        pos = torch.arange(past_len, device = past.device)
        act = pos > lo
        ids = past[act]
        dist = (past_len - pos[act]).to(torch.float32)
        if decay > 0:
            f = (1.0 - (dist - sustain) / decay).clamp_(0.0, 1.0)
        else:
            f = torch.ones_like(dist)
        f = torch.where(dist <= sustain, torch.ones_like(dist), f)
        valid = (ids >= 0) & (ids < vocab)
        factors.scatter_reduce_(0, ids[valid], f[valid], "amax", include_self = True)
    f = factors + 1e-30
    f1 = (1.0 - f) + 1e-30
    w = torch.where(logits > 0.0, logits / spec.rep_p, logits * spec.rep_p)
    return logits * f1 + w * f


def _truncate(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """
    Torch mirror of the discrete SS_TopK -> SS_Normalize -> SS_TopP sequence (same order
    and threshold semantics as the fused kernel's histogram select): top_k keeps exactly
    the K highest logits; top_p then normalizes over the kept set and keeps sorted
    positions while the cumulative probability stays <= top_p (position 0 always kept).
    Works on (..., vocab) — 1D rows and batched row stacks alike.
    """
    vocab = logits.shape[-1]
    if 0 < top_k < vocab:
        s_logits, s_idx = torch.sort(logits, dim = -1, descending = True)
        kept = torch.full_like(s_logits, -float("inf"))
        kept[..., :top_k] = s_logits[..., :top_k]
        logits = torch.full_like(logits, -float("inf")).scatter(-1, s_idx, kept)
    if 0.0 < top_p < 1.0:
        s_logits, s_idx = torch.sort(logits, dim = -1, descending = True)
        probs = torch.softmax(s_logits, dim = -1)
        keep = probs.cumsum(dim = -1) <= top_p
        keep[..., 0] = True
        s_logits = torch.where(keep, s_logits, torch.full_like(s_logits, -float("inf")))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, s_idx, s_logits)
    return logits


def spec_transform(
    logits_row: torch.Tensor,
    past_ids: torch.Tensor | None,
    spec: SpecTransform,
) -> torch.Tensor:
    """
    Target-side post-transform distribution p for one verification position, as fp32
    log-probs over the full vocabulary. Mirrors the job sampler's executed stack:
    rep penalty (over past_ids) -> [temperature] -> top_k -> top_p -> [temperature] ->
    log_softmax, with temperature applied before the filters iff temp_first.
    """
    logits = logits_row.float().clone()
    if spec.rep_p != 1.0 and past_ids is not None and past_ids.numel():
        logits = _apply_rep_penalty(logits, past_ids.to(logits.device), spec)
    if spec.temp_first and spec.temperature != 1.0:
        logits = logits / spec.temperature
    logits = _truncate(logits, spec.top_k, spec.top_p)
    if not spec.temp_first and spec.temperature != 1.0:
        logits = logits / spec.temperature
    return torch.log_softmax(logits, dim = -1)


def spec_transform_batched(
    logits_rows: torch.Tensor,
    pasts: list,
    spec: SpecTransform,
) -> torch.Tensor:
    """
    Batched spec_transform: (w, vocab) logits rows -> (w, vocab) fp32 log-probs. pasts[i]
    is the past-ids row for position i (None disables the penalty for that row). The
    rep penalty is applied per row (its factors depend on each row's own past length);
    temperature and the top_k/top_p truncation batch over rows. Launch-only, no syncs —
    the spec verify path builds all rows optimistically and reads back once per window.
    """
    logits = logits_rows.float()
    if spec.rep_p != 1.0:
        dev = logits.device
        logits = torch.stack([
            _apply_rep_penalty(logits[i], p.to(dev), spec)
            if p is not None and p.numel() else logits[i]
            for i, p in enumerate(pasts)
        ])
    if spec.temp_first and spec.temperature != 1.0:
        logits = logits / spec.temperature
    logits = _truncate(logits, spec.top_k, spec.top_p)
    if not spec.temp_first and spec.temperature != 1.0:
        logits = logits / spec.temperature
    return torch.log_softmax(logits, dim = -1)


def shape_q_logprobs(
    logq: torch.Tensor,
    temperature: float = 0.6,
    top_k: int = 20,
    top_p: float = 0.95,
) -> torch.Tensor:
    """
    q shaping (design §8.4): rescale the draft head's log-probs by temperature and apply
    q's OWN top_k/top_p truncation (its own statistics only, no peek at p). Exact
    speculative sampling is unbiased for any q, so this costs nothing in correctness.
    """
    logits = logq / temperature
    logits = _truncate(logits, top_k, top_p)
    return torch.log_softmax(logits, dim = -1)


def shadow_overlaps(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """
    Expected per-position acceptance E[sum_x min(p(x), q(x))] for the four q variants:
    raw (T=1.0), T=0.8, T=0.6, and T=0.6 + q-shaped top_k=20/top_p=0.95. q lives on the
    pruned-head slice (first n2 columns) and is zero outside it, so the sum only needs
    the slice; p is restricted to the same columns. Returns a (4,) fp32 tensor.
    """
    n2 = logq.shape[-1]
    p = logp[:n2].exp()
    ovs = []
    for t in (1.0, 0.8, 0.6):
        lq = logq if t == 1.0 else torch.log_softmax(logq / t, dim = -1)
        ovs.append(torch.minimum(p, lq.exp()).sum())
    lq = shape_q_logprobs(logq, 0.6, 20, 0.95)
    ovs.append(torch.minimum(p, lq.exp()).sum())
    return torch.stack(ovs)


def job_spec(job) -> SpecTransform | None:
    """
    M2 spec-path eligibility for one job (design §4): the job's sampler stack must be in
    the supported subset AND nothing may make the per-position distribution depend on
    state the spec path does not model (filters, forced tokens, logit masks, prob
    exports, multi-sequence/CFG, token healing). Returns the transform, or None to keep
    the job on the legacy match-verify path.
    """
    if (
        len(job.sequences) != 1 or
        job.filters or
        job.forced_ids is not None or
        job.return_probs or
        job.return_top_tokens > 0 or
        job.device_logit_mask is not None or
        job.new_tokens < 0
    ):
        return None
    return extract_spec(job.sampler)


def shaped_q(logq: torch.Tensor) -> torch.Tensor:
    """
    The actual draft distribution q used by the spec path (M1-measured shaping: temp 0.6
    + the draft head's own top_k=20/top_p=0.95 truncation). Drafting samples from this
    and verification must test against exactly the same q.
    """
    return shape_q_logprobs(logq, SPEC_Q_TEMP, SPEC_Q_TOP_K, SPEC_Q_TOP_P)


def q_sample(logq_row: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Draw one draft token from the shaped q (deterministic given the seed; the caller
    seeds from job.rng so runs stay reproducible per job seed).
    """
    probs = shaped_q(logq_row).exp()
    gen = torch.Generator(device = logq_row.device)
    gen.manual_seed(seed)
    return torch.multinomial(probs, 1, generator = gen)


def spec_accept(logu: float, logpd: float, logqd: float) -> bool:
    """
    Acceptance test in log space (design §8: avoids overflow for tiny q(d)): accept the
    draft iff log u < min(0, logp(d) - logq(d)). q(d) == 0 (draft outside the pruned-head
    slice) is passed as logqd = -inf and always accepts; p(d) == 0 gives logpd = -inf and
    always rejects.
    """
    return logu < min(0.0, logpd - logqd)


def spec_residual_probs(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """
    Rejection resampling distribution renorm((p - q)+) over the full vocabulary. q is
    zero outside the pruned-head slice, so the residual there equals p without
    materializing q beyond the slice. Falls back to p in the fp-degenerate p == q case
    (unreachable via rejection in exact math, since acceptance is then 1).
    """
    n2 = logq.shape[-1]
    p = logp.exp()
    r = p.clone()
    r[:n2] = (p[:n2] - logq.exp()).clamp_(min = 0.0)
    s = r.sum()
    if s <= 0.0:
        return p
    return r / s
