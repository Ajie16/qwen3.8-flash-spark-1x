import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import random
import torch

import exllamav3.generator.sampler.custom as sampler_custom
from exllamav3.generator.sampler.custom import (
    CustomSampler,
    SamplingState,
    SS_RepP,
    SS_Temperature,
    SS_TopK,
    SS_TopP,
    SS_MinP,
    SS_Sample,
)
from exllamav3.generator.spec_sampling import extract_spec, spec_transform

"""
Parity test for spec_transform (M1 shadow mode, docs/spec_sampling.md §8.1): the
standalone target-side transform must reproduce the distribution of the executed sampler
stack. Reference is the discrete step stack (fusion disabled), sampled hundreds of
thousands of times; the empirical token frequencies are compared against the theoretical
spec_transform distribution with a chi-square statistic over the support.

Runs on cuda (the rep-penalty reference is a CUDA kernel) with small tensors. Standalone:

    python tests/test_spec_transform_parity.py
"""

device = "cuda:0"


def empirical_counts(sampler, logits, n_samples, batch):
    counts = torch.zeros(logits.shape[-1], dtype = torch.long, device = device)
    lb = torch.empty((batch, logits.shape[-1]), dtype = logits.dtype, device = device)
    done = 0
    while done < n_samples:
        # Sampler steps can transform their input in place (fp32 temperature), so every
        # draw starts from a fresh copy of the reference logits
        lb.copy_(logits)
        rand_u32 = random.randrange(0, 1 << 32)
        s = sampler.forward(lb, None, rand_u32, None)
        counts += torch.bincount(s.view(-1), minlength = logits.shape[-1])
        done += batch
    return counts


def apply_rep_penalty_ref(rep_step, logits, past_ids):
    # The rep-penalty kernel is bsz-1 only, so the reference applies the real SS_RepP
    # step once up front (it is deterministic given logits + past) and batch-samples the
    # remaining tail below — an exact reference for the full stack
    state = SamplingState(
        rand_u32 = 0, bsz = 1, dim = logits.shape[-1],
        in_logits = logits.view(1, -1), past_ids = past_ids,
    )
    rep_step.run(state)
    return state.logits.view(-1)


def check_case(name, steps, vocab, n_samples, batch, past_len, seed):
    # Discrete reference stack (fusion disabled); restored after construction
    prev = sampler_custom.fused_sampler_enable
    sampler_custom.fused_sampler_enable = False
    try:
        sampler = CustomSampler(steps)
        rep_step = steps[0] if type(steps[0]) is SS_RepP else None
        tail_sampler = CustomSampler(steps[1:] if rep_step is not None else steps)
    finally:
        sampler_custom.fused_sampler_enable = prev

    spec = extract_spec(sampler)
    assert spec is not None, f"{name}: stack should be in the supported subset"

    torch.manual_seed(seed)
    random.seed(seed)
    logits = (torch.randn(vocab, device = device) * 3.0).half()
    past = torch.randint(0, vocab, (1, past_len), device = device)

    ref_logits = apply_rep_penalty_ref(rep_step, logits, past) if rep_step is not None else logits
    counts = empirical_counts(tail_sampler, ref_logits, n_samples, batch)

    logp = spec_transform(logits, past, spec)
    p = logp.exp().double()
    assert abs(p.sum().item() - 1.0) < 1e-3, f"{name}: p not normalized: {p.sum().item()}"

    support = p > 0
    n_support = int(support.sum())
    outside = int(counts[~support].sum())
    assert outside == 0, f"{name}: {outside} samples outside the theoretical support"

    pc = p[support]
    pc = pc / pc.sum()
    cc = counts[support].double()
    expected = pc * n_samples
    # Bins with negligible expectation make Pearson's statistic invalid; evaluate on the
    # well-populated bins, renormalized (the outside-support check above covers the rest)
    big = expected >= 5.0
    pb = pc[big]
    pb = pb / pb.sum()
    cb = cc[big]
    nb = cb.sum()
    eb = pb * nb
    chi2 = float(((cb - eb) ** 2 / eb).sum())
    df = int(big.sum()) - 1
    emp = cb / cb.sum()
    kl = float((pb * (pb.log() - emp.clamp(min = 1e-300).log())).sum())  # KL(theory || empirical)
    tv = float((pb - emp).abs().sum() / 2)

    limit = df + 6.0 * math.sqrt(2.0 * df)
    print(
        f"{name}: n={n_samples} support={n_support} bins={int(big.sum())} "
        f"chi2={chi2:.1f} df={df} chi2/df={chi2 / max(df, 1):.3f} "
        f"KL(theo||emp)={kl:.2e} TV={tv:.2e} (limit chi2<{limit:.1f})"
    )
    assert chi2 < limit, f"{name}: chi-square {chi2:.1f} exceeds {limit:.1f}"
    assert kl < 2e-3, f"{name}: KL {kl} too large"


def test_parity_deployed_preset():
    # Live preset: temp 0.6 / top_p 0.95 / top_k 20 / rep_penalty 1.05, temp-first
    check_case(
        "deployed(temp0.6,topp0.95,topk20,repp1.05,decay)",
        [SS_RepP(1.05, 2048, 512), SS_Temperature(0.6), SS_TopK(20), SS_TopP(0.95), SS_Sample()],
        vocab = 32768, n_samples = 200_000, batch = 1024, past_len = 3000, seed = 1234,
    )


def test_parity_temp_last():
    check_case(
        "temp-last(topk50,topp0.9,temp0.8,repp1.1)",
        [SS_RepP(1.1, 100000, 0), SS_TopK(50), SS_TopP(0.9), SS_Temperature(0.8), SS_Sample()],
        vocab = 32768, n_samples = 100_000, batch = 1024, past_len = 500, seed = 5678,
    )


def test_parity_temperature_only():
    check_case(
        "temperature-only(temp0.7)",
        [SS_Temperature(0.7), SS_Sample()],
        vocab = 16384, n_samples = 200_000, batch = 1024, past_len = 0, seed = 42,
    )


def test_extract_spec_fused_matches_discrete():
    # The same stack fused (default) must extract the same spec as the discrete form
    def build():
        return CustomSampler([SS_RepP(1.05, 2048, 512), SS_Temperature(0.6),
                              SS_TopK(20), SS_TopP(0.95), SS_Sample()])

    prev = sampler_custom.fused_sampler_enable
    try:
        sampler_custom.fused_sampler_enable = True
        fused_spec = extract_spec(build())
        sampler_custom.fused_sampler_enable = False
        disc_spec = extract_spec(build())
    finally:
        sampler_custom.fused_sampler_enable = prev
    assert fused_spec is not None and disc_spec is not None
    for f in ("temperature", "top_k", "top_p", "temp_first", "rep_p", "sustain_range", "decay_range"):
        a, b = getattr(fused_spec, f), getattr(disc_spec, f)
        assert a == b or math.isclose(a, b), f"fused {fused_spec} != discrete {disc_spec} (field {f})"

    # temp-last form
    prev = sampler_custom.fused_sampler_enable
    try:
        sampler_custom.fused_sampler_enable = True
        s = extract_spec(CustomSampler([SS_TopK(50), SS_TopP(0.9), SS_Temperature(0.8), SS_Sample()]))
    finally:
        sampler_custom.fused_sampler_enable = prev
    assert s is not None and not s.temp_first and abs(s.temperature - 0.8) < 1e-6
    assert s.top_k == 50 and abs(s.top_p - 0.9) < 1e-6
    print("extract_spec fused/discrete parity ok")


def test_extract_spec_unsupported():
    prev = sampler_custom.fused_sampler_enable
    try:
        for fused in (True, False):
            sampler_custom.fused_sampler_enable = fused
            # min_p is outside the v1 subset
            assert extract_spec(CustomSampler([SS_Temperature(0.6), SS_MinP(0.1), SS_Sample()])) is None
            # pure argmax (greedy) is degenerate -> legacy
            from exllamav3.generator.sampler.custom import SS_Argmax
            assert extract_spec(CustomSampler([SS_TopK(20), SS_Argmax()])) is None
    finally:
        sampler_custom.fused_sampler_enable = prev
    print("extract_spec unsupported-stack fallbacks ok")


def test_transform_deterministic():
    # Hand-computed transforms (CPU, no kernels): pins the exact top_k/top_p threshold
    # semantics (position 0 always kept; token kept while cumsum <= top_p), temperature
    # placement and the rep-penalty mirror against known values
    from exllamav3.generator.spec_sampling import SpecTransform

    ni = -float("inf")

    # top_k then top_p: cumsum including the token must stay <= top_p
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    logp = spec_transform(logits, None, SpecTransform(top_k = 3, top_p = 0.9))
    # softmax([4,3,2]) = [0.6652, 0.2447, 0.0900]; cumsum[1] = 0.9099 > 0.9 -> only token 0
    expect = torch.tensor([0.0, ni, ni, ni, ni, ni])
    assert torch.equal(logp, expect), f"top_k/top_p: {logp}"

    # temp-first: filters see the tempered distribution
    logits = torch.tensor([2.0, 1.0, 0.0])
    logp = spec_transform(logits, None, SpecTransform(temperature = 2.0, top_p = 0.85, temp_first = True))
    # tempered [1, 0.5, 0]: cumsum = [0.5065, 0.8137, 1.0] -> drop last, renorm over first two
    expect = torch.log_softmax(torch.tensor([1.0, 0.5, ni]), dim = -1)
    assert torch.allclose(logp, expect), f"temp-first: {logp} vs {expect}"

    # temp-last: filters see the untempered distribution
    logp = spec_transform(logits, None, SpecTransform(temperature = 2.0, top_p = 0.85, temp_first = False))
    # untempered cumsum = [0.6652, 0.9099, 1.0] -> keep only token 0
    expect = torch.tensor([0.0, ni, ni])
    assert torch.equal(logp, expect), f"temp-last: {logp}"

    # rep penalty mirror: positive logits divided, negative multiplied, non-past untouched
    logits = torch.tensor([2.0, -2.0, 4.0])
    past = torch.tensor([0, 1])
    logp = spec_transform(logits, past, SpecTransform(rep_p = 2.0, sustain_range = 1000, decay_range = 0))
    expect = torch.log_softmax(torch.tensor([1.0, -4.0, 4.0]), dim = -1)
    assert torch.allclose(logp, expect, atol = 1e-6), f"rep_p: {logp} vs {expect}"

    print("deterministic transform checks ok")


def test_shadow_observe_smoke():
    # Drives Generator._spec_shadow_observe with stub jobs. Covers the two failure modes
    # seen outside the stub world: (a) sequence_ids.torch() is (1, seq_len) — 2D, the same
    # shape the live sampler gets as past_ids — and the draft continuation must align with
    # it for positions past the legacy cut; (b) any exception inside the observation must
    # be contained (rate-limited warning, window skipped), never propagated into
    # generation.
    import logging
    from types import SimpleNamespace
    from exllamav3.generator.generator import Generator, logger
    from exllamav3.generator.sampler.custom import SS_Fused

    infos, warnings = [], []
    class H(logging.Handler):
        def emit(self, r):
            (warnings if r.levelno >= logging.WARNING else infos).append(r.getMessage())
    h = H()
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    try:
        vocab, window = 64, 3
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        def make_job(seq_ids_obj):
            return SimpleNamespace(
                sequences = [SimpleNamespace(sequence_ids = seq_ids_obj)],
                filters = [], forced_ids = None, return_probs = False, return_top_tokens = 0,
                device_logit_mask = None, new_tokens = 1,
                sampler = SimpleNamespace(steps = [SS_Fused(SS_Fused.MODE_SAMPLE, 1.0)]),
            )

        class FakeSeqIds:
            def torch(self): return torch.zeros((1, 10), dtype = torch.long)  # real 2D shape

        g = SimpleNamespace(
            active_jobs = [make_job(FakeSeqIds())],
            _spec_shadow_stats = None, _spec_shadow_errors = 0,
            tokenizer = SimpleNamespace(actual_vocab_size = vocab),
        )
        # The real Generator has this as a method; the wrapper calls it on self
        g._spec_shadow_observe_inner = Generator._spec_shadow_observe_inner.__get__(g)
        batch_logits = torch.randn(1, window + 1, vocab, device = device)
        draft_tokens = torch.zeros(1, window, dtype = torch.long)
        # accepted_length = 1 -> k = 0, so positions 1..window-1 take the cat path
        for _ in range(120):
            g._spec_shadow_logq = [
                torch.log_softmax(torch.randn(1, 32, device = device), dim = -1)
                for _ in range(window)
            ]
            Generator._spec_shadow_observe(g, batch_logits, draft_tokens, [0, 1], [1], set())
        assert len(infos) == 2, infos
        assert infos[0].startswith("spec-shadow: [interval] windows=50 "), infos[0]
        assert "| [total] windows=50 " in infos[0]
        assert "| [total] windows=100 " in infos[1]

        # Exception containment: a job whose state access raises is skipped with a
        # rate-limited warning; nothing propagates, stats are untouched
        class BadSeqIds:
            def torch(self): raise RuntimeError("boom")
        g.active_jobs = [make_job(BadSeqIds())]
        g._spec_shadow_logq = [
            torch.log_softmax(torch.randn(1, 32, device = device), dim = -1)
            for _ in range(window)
        ]
        stats_before = dict(g._spec_shadow_stats)
        Generator._spec_shadow_observe(g, batch_logits, draft_tokens, [0, 1], [1], set())
        assert g._spec_shadow_errors == 1
        assert len(warnings) == 1 and "spec-shadow" in warnings[0] and "boom" in warnings[0], warnings
        assert g._spec_shadow_stats["windows"] == stats_before["windows"]
    finally:
        logger.removeHandler(h)
    print("shadow observe smoke ok")


if __name__ == "__main__":
    test_transform_deterministic()
    test_extract_spec_fused_matches_discrete()
    test_extract_spec_unsupported()
    test_shadow_observe_smoke()
    test_parity_deployed_preset()
    test_parity_temp_last()
    test_parity_temperature_only()
    print("all spec_transform parity checks passed")
