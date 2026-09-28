import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import torch
from types import SimpleNamespace

from exllamav3.generator.generator import Generator, _hybrid_pad_rows
from exllamav3.generator.job import Job
from exllamav3.ext import exllamav3_ext as ext

"""
M3 hybrid ngram+MTP drafting (EXL3_HYBRID_NGRAM) — CPU gates. All CPU, no model, no GPU.

- test_pad_rows: mixed-width assembly, pad token, None rows.
- test_sam_intermittent_parity: BC_SAM.accept_tensor consumes history incrementally, so
  probing every few tokens (ngram this round, MTP the next) must track one-shot feeding.
- test_sam_rewind_rebuild: a shrinking sequence (banned-string rewind) rebuilds the
  automaton instead of corrupting it.
- test_probe_gate: Job.probe_ngram_draft match-length gate and continuation capping.
- test_hybrid_selection: Generator.iterate_hybrid_gen source choice — pure-MTP fallback,
  pure-ngram (no drafter forward, stale draft state cleared), mixed (row overwrite,
  spec suppression, calibrator-skip set).
- test_verify_walk_with_pads / test_hybrid_account: pad positions verify as ordinary
  draft positions, and per-source accounting clamps pad-luck accepts to the real width.

Standalone: python tests/test_hybrid_draft_cpu.py
"""


class _SeqStub:
    def __init__(self, ids):
        self.sequence_ids = SimpleNamespace(torch = lambda: ids.view(1, -1))


def _job_stub(ids):
    return SimpleNamespace(sam = ext.BC_SAM(), sequences = [_SeqStub(ids)])


def _gen_stub():
    g = Generator.__new__(Generator)
    g.hybrid_ngram = True
    g.hybrid_ngram_min_match = 4
    g.hybrid_ngram_max = 8
    g._hybrid_round_src = None
    g._hybrid_ngram_jobs = set()
    g._hybrid_suppress_spec = False
    g._hybrid_stats = {
        "rounds": 0, "ng_rounds": 0,
        "ng_drafted": 0, "ng_accepted": 0,
        "mtp_drafted": 0, "mtp_accepted": 0,
    }
    g._spec_shadow_logq = None
    g._spec_draft_specs = None
    g._draft_conf_round = None
    return g


def _sel_job(cand):
    # Minimal job for iterate_hybrid_gen: prefill done, single sequence, canned probe
    return SimpleNamespace(
        sequences = [object()],
        sam = object() if cand is not None else None,
        is_prefill_done = lambda: True,
        probe_ngram_draft = lambda n, m: cand,
    )


def test_pad_rows():
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[7]])
    out = _hybrid_pad_rows([a, None, b], pad_token = 0)
    assert out.shape == (3, 3)
    assert out[0].tolist() == [1, 2, 3]
    assert out[1].tolist() == [0, 0, 0]
    assert out[2].tolist() == [7, 0, 0]
    assert out.dtype == torch.long
    print("pad rows ok")


def test_sam_intermittent_parity():
    # A sequence ending in a repeated block: noise + block + gap + block + gap + block
    random.seed(1)
    block = [random.randint(1000, 60000) for _ in range(11)]
    seq = [3, 4] + block + [7, 8] + block + [9] + block
    ids = torch.tensor(seq, dtype = torch.long)

    # One-shot
    sam1 = ext.BC_SAM()
    r1 = sam1.accept_tensor(ids)

    # Intermittent: probe every 1-4 tokens (ngram/MTP round mix)
    sam2 = ext.BC_SAM()
    pos = 0
    r2 = (-1, -1)
    while pos < len(seq):
        step = random.randint(1, 4)
        pos = min(pos + step, len(seq))
        r2 = sam2.accept_tensor(ids[:pos])

    assert r1 == r2, (r1, r2)
    assert r1[1] - r1[0] >= len(block), r1
    # The continuation after the earliest occurrence is inside the sequence
    assert 0 <= r1[0] < r1[1] < len(seq)
    print(f"sam intermittent parity ok (match_len={r1[1] - r1[0]})")


def test_sam_rewind_rebuild():
    block = [10, 20, 30, 40, 50]
    ids = torch.tensor(block + [99] + block + [98] + block, dtype = torch.long)
    sam = ext.BC_SAM()
    sam.accept_tensor(ids)
    # Rewind: sequence shrinks (banned-string suppression); automaton must rebuild
    short = ids[:7]
    beg, end = sam.accept_tensor(short)
    assert end - beg >= 0
    # And keep working correctly afterwards
    ids2 = torch.cat((short, torch.tensor(block, dtype = torch.long)))
    beg2, end2 = sam.accept_tensor(ids2)
    assert end2 - beg2 >= 5
    print("sam rewind rebuild ok")


def test_probe_gate():
    block = [101, 102, 103, 104, 105, 106]
    # Sequence ends with a repeat of block; the match is the trailing block and the
    # draft copies what followed the FIRST occurrence: [1, 2] then the block again
    seq = block + [1, 2] + block
    ids = torch.tensor(seq, dtype = torch.long)
    job = _job_stub(ids)
    draft, mlen = Job.probe_ngram_draft(job, 8, 4)
    assert mlen == len(block)
    assert draft[0].tolist() == [1, 2] + block[:6]

    # draft_length caps the continuation
    job = _job_stub(ids)
    draft, _ = Job.probe_ngram_draft(job, 3, 4)
    assert draft[0].tolist() == [1, 2, 101]

    # min_match above the repeat length -> empty draft, match still reported
    job2 = _job_stub(ids)
    draft2, mlen2 = Job.probe_ngram_draft(job2, 8, 100)
    assert draft2.shape[-1] == 0 and mlen2 == len(block)

    # No repeat at all -> empty draft, zero match
    job3 = _job_stub(torch.arange(50, dtype = torch.long))
    draft3, mlen3 = Job.probe_ngram_draft(job3, 8, 4)
    assert draft3.shape[-1] == 0 and mlen3 == 0
    print("probe gate ok")


def test_hybrid_selection():
    g = _gen_stub()
    cand = torch.tensor([[11, 12, 13, 14, 15]])

    calls = []
    def fake_mtp(results):
        calls.append(g._hybrid_suppress_spec)
        g._spec_shadow_logq = ["logq"]
        g._spec_draft_specs = ["specs"]
        g._draft_conf_round = {"ids": None}
        return torch.tensor([[21, 22], [23, 24]])
    g.iterate_draftmodel_mtp_gen = fake_mtp

    # Pure MTP round: no candidates
    g.active_jobs = [_sel_job(None), _sel_job(None)]
    out = Generator.iterate_hybrid_gen(g, [])
    assert calls == [False]
    assert out.tolist() == [[21, 22], [23, 24]]
    assert all(v[0] == "mtp" and v[1] == 2 for v in g._hybrid_round_src.values())
    assert g._hybrid_stats["rounds"] == 1 and g._hybrid_stats["ng_rounds"] == 0

    # Pure ngram round: MTP drafter not called, stale draft state cleared
    g.active_jobs = [_sel_job((cand, 6)), _sel_job((torch.tensor([[31]]), 5))]
    out = Generator.iterate_hybrid_gen(g, [])
    assert calls == [False], "pure ngram round must skip the MTP draft forward"
    assert out.shape == (2, 5)
    assert out[0].tolist() == [11, 12, 13, 14, 15]
    assert out[1].tolist() == [31, 0, 0, 0, 0]
    assert g._spec_shadow_logq is None and g._spec_draft_specs is None
    assert g._draft_conf_round is None
    assert len(g._hybrid_ngram_jobs) == 2
    assert all(v[0] == "ngram" for v in g._hybrid_round_src.values())
    assert g._hybrid_stats["ng_rounds"] == 1

    # Mixed round: MTP runs with spec suppressed, ngram row overwrites, wider window pads
    g._spec_shadow_logq = None
    g.active_jobs = [_sel_job((cand, 6)), _sel_job(None)]
    out = Generator.iterate_hybrid_gen(g, [])
    assert calls == [False, True], "mixed round must suppress spec drafting"
    assert out.shape == (2, 5)
    assert out[0].tolist() == [11, 12, 13, 14, 15]
    assert out[1].tolist() == [23, 24, 0, 0, 0]
    assert g._spec_shadow_logq is None, "shadow rows describe discarded MTP drafts"
    assert len(g._hybrid_ngram_jobs) == 1
    kinds = sorted(v[0] for v in g._hybrid_round_src.values())
    assert kinds == ["mtp", "ngram"]
    assert g._hybrid_suppress_spec is False, "flag must reset after the draft call"

    # Mixed round with no MTP carry (mtp returns None): ngram rows still draft,
    # the other row is all-pad
    g.iterate_draftmodel_mtp_gen = lambda results: None
    g.active_jobs = [_sel_job((cand, 6)), _sel_job(None)]
    out = Generator.iterate_hybrid_gen(g, [])
    assert out.shape == (2, 5)
    assert out[0].tolist() == [11, 12, 13, 14, 15]
    assert out[1].tolist() == [0, 0, 0, 0, 0]
    print("hybrid selection ok")


def test_verify_walk_with_pads():
    # Emulate the legacy match-verify loop over an assembled padded draft: positions
    # accept while the target samples the draft token; the first mismatch rejects the
    # remainder. Pad tokens (0) accept only on a lucky target match.
    drafts = _hybrid_pad_rows([torch.tensor([[5, 6, 7]]), torch.tensor([[9]])])
    target = [5, 6, 42]   # row 0: accept 2, reject at 2; row 1 would accept 1 if target[0]==9
    accepted = 1
    for i in range(drafts.shape[-1]):
        if i >= len(target) or drafts[0, i].item() != target[i]:
            break
        accepted += 1
    assert accepted == 3  # 1 sampled + 2 accepted drafts

    # Lucky pad: target samples 0 at the pad position -> accepted, still exact
    drafts2 = _hybrid_pad_rows([torch.tensor([[9, 0]])])
    target2 = [9, 0]
    accepted2 = 1
    for i in range(drafts2.shape[-1]):
        if i >= len(target2) or drafts2[0, i].item() != target2[i]:
            break
        accepted2 += 1
    assert accepted2 == 3
    print("verify walk with pads ok")


def test_hybrid_account():
    g = _gen_stub()
    j1, j2, j3 = (SimpleNamespace(serial_number = i) for i in range(3))
    g.active_jobs = [j1, j2, j3]
    g._hybrid_round_src = {
        id(j1): ("ngram", 5),
        id(j2): ("mtp", 2),
        id(j3): ("ngram", 4),
    }
    # j3's window was abandoned (rewind) and must be skipped; j1 accepted past its real
    # width via a pad match (accepted_length 7 > width 5 + 1) and must clamp
    logit_mapping = [0, 3, 5, 8]
    accepted_lengths = [7, 2, 3]
    Generator._hybrid_account(g, logit_mapping, accepted_lengths, {id(j3)})
    st = g._hybrid_stats
    assert st["ng_drafted"] == 5 and st["ng_accepted"] == 5, st
    assert st["mtp_drafted"] == 2 and st["mtp_accepted"] == 1, st
    assert g._hybrid_round_src is None
    print("hybrid account ok")


def test_ng_adaptive():
    import exllamav3.generator.generator as G
    old = (G._HYBRID_ADAPT_ROUNDS, G._HYBRID_PROBE_ROUNDS, G._HYBRID_MIN_ACC)
    G._HYBRID_ADAPT_ROUNDS, G._HYBRID_PROBE_ROUNDS, G._HYBRID_MIN_ACC = 4, 3, 0.25
    try:
        g = _gen_stub()
        j = SimpleNamespace(serial_number = 7)
        assert Generator._hybrid_ng_engaged(g, j)  # default on

        # Four ngram rounds accepting 1/16 each (0.0625 < 0.25) -> disengage
        for _ in range(4):
            g.active_jobs = [j]
            g._hybrid_round_src = {id(j): ("ngram", 16)}
            Generator._hybrid_account(g, [0, 9], [2], set())
        assert j._ng_on is False
        assert not Generator._hybrid_ng_engaged(g, j)

        # Re-probe after PROBE_ROUNDS off-rounds
        assert not Generator._hybrid_ng_engaged(g, j)
        assert Generator._hybrid_ng_engaged(g, j)  # 3rd call re-engages
        assert j._ng_on is True

        # Good acceptance keeps it engaged
        j2 = SimpleNamespace(serial_number = 8)
        for _ in range(8):
            g.active_jobs = [j2]
            g._hybrid_round_src = {id(j2): ("ngram", 16)}
            Generator._hybrid_account(g, [0, 9], [14], set())
        assert getattr(j2, "_ng_on", True) is True
    finally:
        G._HYBRID_ADAPT_ROUNDS, G._HYBRID_PROBE_ROUNDS, G._HYBRID_MIN_ACC = old
    print("ng adaptive ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_pad_rows()
    test_sam_intermittent_parity()
    test_sam_rewind_rebuild()
    test_probe_gate()
    test_hybrid_selection()
    test_verify_walk_with_pads()
    test_hybrid_account()
    test_ng_adaptive()
    print("all hybrid-draft CPU gates passed")
