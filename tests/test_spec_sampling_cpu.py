import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import random
import torch
from types import SimpleNamespace

from exllamav3.generator.generator import Generator
from exllamav3.generator.spec_sampling import (
    SpecTransform,
    spec_transform,
    shaped_q,
    q_sample,
    spec_accept,
    spec_residual_probs,
    SPEC_Q_TEMP,
    SPEC_Q_TOP_K,
    SPEC_Q_TOP_P,
)

"""
M2 exact speculative sampling — CPU gates (design §5/§6). All CPU, no model, no GPU.

- test_emitted_distribution_exact: the hard gate. The spec algorithm (draft d ~ shaped q,
  accept iff log u < min(0, logp(d) - logq(d)), reject -> renorm((p - q)+)) is simulated
  vectorized over 200k draws using the same primitives the generator calls; the emitted
  token distribution must be statistically indistinguishable from p (chi-square), and the
  measured acceptance must match E[sum min(p, q)].
- test_residual_edge_cases: p ⟂ q and p == q.
- test_verify_job_bookkeeping: drives Generator._spec_verify_job with a stub job through
  a full-accept window and an immediate-reject window, asserting the emitted-token /
  accepted_length invariant the post-batch MTP carry relies on
  (mtp_last_hidden = target_hidden[accepted_length - 1] must be the last accepted
  position, design §8.2).

Standalone: python tests/test_spec_sampling_cpu.py
"""

device = "cpu"


def _rand_pq(vocab, n2, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(vocab, generator = g) * 2.0
    logp = torch.log_softmax(logits, dim = -1)
    logq_raw = torch.randn(n2, generator = g) * 2.0
    return logp, torch.log_softmax(logq_raw, dim = -1)


def test_emitted_distribution_exact():
    vocab, n2, n = 256, 128, 200_000
    logp, logq_raw = _rand_pq(vocab, n2, seed = 7)
    p = logp.exp()
    logq = shaped_q(logq_raw)
    q = logq.exp()

    gen = torch.Generator().manual_seed(1234)
    # Draft d ~ q, accept with min(1, p(d)/q(d)) via the generator's log-space test,
    # reject -> renorm((p - q)+); all vectorized, same math as Generator._spec_verify_job
    d = torch.multinomial(q, n, replacement = True, generator = gen)
    logu = torch.rand(n, generator = gen).log()
    thresh = torch.minimum(torch.zeros(n), logp[d] - logq[d])
    accept = logu < thresh
    residual = spec_residual_probs(logp, logq)
    r = torch.multinomial(residual, n, replacement = True, generator = gen)
    emitted = torch.where(accept, d, r)

    # Measured acceptance vs the theoretical total-variation overlap
    overlap = float(torch.minimum(p[:n2], q).sum())
    acc = float(accept.float().mean())
    print(f"acceptance={acc:.4f} vs E[sum min(p,q)]={overlap:.4f}")
    assert abs(acc - overlap) < 0.01

    # Chi-square of emitted tokens against p over the full vocabulary
    counts = torch.bincount(emitted, minlength = vocab).double()
    expected = p.double() * n
    big = expected >= 5.0
    chi2 = float(((counts[big] - expected[big]) ** 2 / expected[big]).sum())
    df = int(big.sum()) - 1
    limit = df + 6.0 * math.sqrt(2.0 * df)
    print(f"exactness: n={n} bins={int(big.sum())} chi2={chi2:.1f} df={df} "
          f"chi2/df={chi2 / df:.3f} (limit {limit:.1f})")
    assert chi2 < limit, f"emitted distribution differs from p: chi2={chi2:.1f}"


def test_residual_edge_cases():
    vocab, n2 = 64, 32

    # p ⟂ q: q one-hot on token 5 (top_k=1 shaping), p has zero mass there -> every
    # draft is rejected and the residual is exactly p
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(vocab, generator = g)
    logits[5] = -float("inf")
    logp = torch.log_softmax(logits, dim = -1)
    logq = torch.log_softmax(torch.randn(n2, generator = g), dim = -1)
    logq = torch.where(
        torch.arange(n2) == 5, torch.zeros(n2), torch.full((n2,), -float("inf"))
    )
    assert not spec_accept(-10.0, float(logp[5]), float(logq[5]))  # p(d) = 0 -> reject
    residual = spec_residual_probs(logp, logq)
    assert torch.allclose(residual, logp.exp(), atol = 1e-6)

    # q outside the pruned-head slice is zero: logqd = -inf accepts with probability 1
    assert spec_accept(-1e30, float(logp[40]), -float("inf"))

    # p == q as distributions (p's mass entirely inside the slice): acceptance is 1
    # (log u < 0 for every u in [0, 1)) and the degenerate residual falls back to p
    g2 = torch.Generator().manual_seed(4)
    logits2 = torch.randn(vocab, generator = g2)
    logits2[n2:] = -float("inf")
    logp2 = torch.log_softmax(logits2, dim = -1)
    logq2 = logp2[:n2].clone()
    for u in (0.0, 1e-9, 0.3, 0.9, 0.999999):
        assert spec_accept(math.log(u) if u > 0 else -float("inf"), float(logp2[7]), float(logq2[7]))
    residual = spec_residual_probs(logp2, logq2)
    assert torch.allclose(residual, logp2.exp(), atol = 1e-6)
    print("residual edge cases ok")


class _StubJob:
    """
    Minimal Job stand-in for _spec_verify_job. receive_sample appends the token to the
    recorded emission log (the real one also advances sequence_ids / kv_position, which
    the MTP carry then reads back via accepted_lengths).
    """
    def __init__(self, bonus_token):
        self.sequences = [object()]
        self.rng = random.Random(0)
        self.current_device_ids = None
        self.rejected_draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.checkpoint_rewound = False
        self.emitted = []
        self._bonus = bonus_token

    def receive_logits(self, token_logits):
        return torch.tensor([[self._bonus]], dtype = torch.long), None, None, None

    def receive_sample(self, token_logits, next_token, next_k_tokens, next_k_probs, next_prob, results):
        self.emitted.append(int(next_token.view(-1)[0]))
        return False, next_token, False

    def prepare_logit_mask(self):
        pass

    def prepare_sampling_past_ids(self):
        pass


def _run_verify_window(job, logp_row, logq_rows_list, draft_tokens_row):
    vocab = logp_row.shape[-1]
    window = len(logq_rows_list)
    job_logits = logp_row.view(1, 1, vocab).repeat(1, window + 1, 1)
    draft_tokens = draft_tokens_row.view(1, window)
    g = SimpleNamespace(
        _spec_draft_specs = [SpecTransform()],  # temp 1.0, no penalties
        _spec_shadow_logq = [row.view(1, -1) for row in logq_rows_list],
        tokenizer = SimpleNamespace(actual_vocab_size = vocab),
    )
    reject_calls = []
    def fake_reject(job_, j_, i_, bs_):
        reject_calls.append((j_, i_))
        job_.rejected_draft_tokens += job_logits.shape[1] - 1 - i_
        return job_logits.shape[1] - 1 - i_
    accepted_length, rejected = Generator._spec_verify_job(
        g, job, 0, job_logits, draft_tokens, None, [], set(), [], [], fake_reject,
    )
    return accepted_length, rejected, reject_calls


def test_verify_job_bookkeeping():
    vocab, n2, window = 64, 32, 4
    logp, logq_raw = _rand_pq(vocab, n2, seed = 11)
    logq = shaped_q(logq_raw)
    drafts = torch.tensor([int(logq.argmax())] * window, dtype = torch.long)

    # Full-accept window: p == q over the slice (p(d) == q(d) -> thresh 0 -> accept
    # every position), then a bonus token from the real sampler call
    same = torch.full((vocab,), -float("inf"))
    same[:n2] = logq
    job = _StubJob(bonus_token = 61)
    accepted_length, rejected, reject_calls = _run_verify_window(
        job, same, [logq_raw] * window, drafts
    )
    assert rejected == 0 and not reject_calls
    assert job.emitted == drafts.tolist() + [61], job.emitted
    assert job.accepted_draft_tokens == window
    # MTP carry invariant (design 8.2): accepted_length must equal the number of tokens
    # consumed this round, so target_hidden[accepted_length - 1] is the last accepted
    # position and the draft-cache realignment re-prefills exactly the accepted drafts
    assert accepted_length == len(job.emitted) == window + 1

    # Immediate reject: p(d_0) = 0 -> deterministic rejection at position 0, one
    # residual token emitted, remainder rejected from position 0
    block = logp.clone()
    block[int(drafts[0])] = -float("inf")
    block = torch.log_softmax(block, dim = -1)
    job = _StubJob(bonus_token = 61)
    accepted_length, rejected, reject_calls = _run_verify_window(
        job, block, [logq_raw] * window, drafts
    )
    assert reject_calls == [(0, 0)]
    assert rejected == window
    assert len(job.emitted) == 1 and job.emitted[0] != int(drafts[0])
    assert job.accepted_draft_tokens == 0
    assert accepted_length == len(job.emitted) == 1
    print("verify-job bookkeeping ok (full-accept and immediate-reject windows)")


def test_q_sample_determinism():
    _, logq_raw = _rand_pq(64, 32, seed = 5)
    a = q_sample(logq_raw, 42)
    b = q_sample(logq_raw, 42)
    assert int(a[0]) == int(b[0])
    # draws stay inside the shaped support
    support = shaped_q(logq_raw) > -float("inf")
    for seed in range(50):
        assert bool(support[int(q_sample(logq_raw, seed)[0])])
    print("q_sample determinism/support ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_residual_edge_cases()
    test_q_sample_determinism()
    test_verify_job_bookkeeping()
    test_emitted_distribution_exact()
    print("all spec-sampling CPU gates passed")
