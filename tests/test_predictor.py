import numpy as np
import torch
import torch.nn.functional as F

from tfpose import rotations as rot
from tfpose import tesseract as T
from tfpose.features import assert_frozen, freeze
from tfpose.predictor import HierMLP, parent_centres


def _greedy(model, x, depth):
    h = model.encode(x)
    chart = model.root(h).argmax(-1)
    path = torch.zeros(len(x), 0, dtype=torch.long)
    for l in range(depth):
        pq = torch.as_tensor(T.decode(chart.numpy(), path.numpy()), dtype=h.dtype)
        c = model.child_logits(h, torch.full((len(x),), l), pq).argmax(-1)
        path = torch.cat([path, c[:, None]], 1)
    return chart, path


def test_beam1_equals_greedy_and_beams_sorted():
    torch.manual_seed(0)
    m = HierMLP(16, 3, hidden=32, emb=16).eval()
    x = torch.randn(50, 16)
    c, p, s, nodes = m.beam_search(x, beam=1)
    gc, gp = _greedy(m, x, 3)
    assert torch.equal(c[:, 0], gc) and torch.equal(p[:, 0], gp)
    assert nodes == 4 + 3 * 8
    c8, p8, s8, n8 = m.beam_search(x, beam=8)
    assert torch.all(s8[:, :-1] >= s8[:, 1:])
    assert torch.all(s8[:, 0] >= s[:, 0] - 1e-5)  # beam can only improve the path score
    assert n8 == 4 + 4 * 8 + 8 * 8 + 8 * 8


def test_parent_centres_and_teacher_forcing_consistency():
    q = rot.random_quats(100, np.random.default_rng(0))
    chart, path = T.encode(q, 3)
    pq = parent_centres(chart, path)
    assert np.allclose(pq[:, 0], np.eye(4)[chart])
    assert np.allclose(pq[:, 2], T.decode(chart, path[:, :2]))
    m = HierMLP(8, 3, hidden=16, emb=8)
    loss, parts = m.loss(torch.randn(100, 8), torch.tensor(chart), torch.tensor(path), torch.tensor(pq, dtype=torch.float32))
    assert len(parts) == 4 and torch.isfinite(loss)


def test_freeze_asserts():
    lin = freeze(torch.nn.Linear(3, 3))
    assert_frozen(lin)
    lin.weight.requires_grad_(True)
    try:
        assert_frozen(lin)
        raise RuntimeError("should have failed")
    except AssertionError:
        pass


def test_decode_torch_matches_numpy():
    from tfpose.predictor import decode_torch
    for L in (0, 1, 3, 5):
        chart, path = T.all_cells(L) if L <= 3 else T.encode(rot.random_quats(3000, np.random.default_rng(1)), L)
        a = decode_torch(torch.as_tensor(chart), torch.as_tensor(path)).double().numpy()
        b = T.decode(chart, path)
        assert np.allclose(a, b, atol=1e-6)


def test_beam_search_fast_matches_reference():
    from tfpose.predictor import beam_search_fast
    torch.manual_seed(1)
    m = HierMLP(16, 4, hidden=32, emb=16).eval()
    x = torch.randn(40, 16)
    for beam in (1, 4):
        c1, p1, s1, _ = m.beam_search(x, beam=beam)
        c2, p2, s2 = beam_search_fast(m, x, beam=beam)
        assert torch.equal(c1, c2) and torch.equal(p1, p2) and torch.allclose(s1, s2, atol=1e-5)
