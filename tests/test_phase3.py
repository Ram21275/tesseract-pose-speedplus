import numpy as np
import torch

from tfpose import phase3 as P, rotations as rot, tesseract as T
from tfpose.predictor import decode_torch, parent_centres

RNG = np.random.default_rng(0)


def test_rotvec_roundtrip():
    v = torch.as_tensor(RNG.normal(size=(500, 3)) * 0.8, dtype=torch.float64)
    v = v[v.norm(dim=-1) < 3.0]
    assert torch.allclose(P.quat_to_rotvec(P.rotvec_to_quat(v)), v, atol=1e-9)


def test_residual_target_recovers_gt():
    q = rot.random_quats(2000, RNG)
    c, p = T.encode(q, 5)
    d = torch.as_tensor(P.leaf_rotvec_targets(q, c, p), dtype=torch.float64)
    qc = torch.as_tensor(T.decode(c, p), dtype=torch.float64)
    qr = P.qmul(qc, P.rotvec_to_quat(d)).numpy()
    assert np.degrees(rot.geodesic_distance(qr, q)).max() < 1e-3
    assert np.degrees(d.norm(dim=-1).numpy()).max() < 6.3        # within the exact L5 cell bound (6.2 deg)


def test_soft_targets_valid():
    q = rot.random_quats(3000, RNG)
    c, p = T.encode(q, 5)
    root, child = P.soft_targets(q, c, p)
    assert np.allclose(root.sum(1), 1, atol=1e-5) and np.allclose(child.sum(2), 1, atol=1e-5)
    assert np.mean(root.argmax(1) == c) > 0.99                    # nearest chart centre = assigned chart
    assert np.mean(child.argmax(2) == p) > 0.8


def test_gru_beam1_equals_greedy():
    torch.manual_seed(0)
    m = P.HierGRU(16, 3, hidden=32, emb=8, gru=16).eval()
    x = torch.randn(30, 16)
    c, p, s = m.beam_search(x, beam=1)
    h = m.encode(x); chart = m.root(h).argmax(-1); st = m.init_state(h, chart)
    prev = torch.full((30,), 8, dtype=torch.long); path = torch.zeros(30, 0, dtype=torch.long)
    for l in range(3):
        lg, st = m.step(h, st, chart, prev, torch.full((30,), l), decode_torch(chart, path))
        prev = lg.argmax(-1); path = torch.cat([path, prev[:, None]], 1)
    assert torch.equal(c[:, 0], chart) and torch.equal(p[:, 0], path)
    c8, p8, s8 = m.beam_search(x, beam=8)
    assert torch.all(s8[:, :-1] >= s8[:, 1:]) and torch.all(s8[:, 0] >= s[:, 0] - 1e-5)


def test_gru_teacher_forcing_shapes():
    m = P.HierGRU(16, 4, hidden=32, emb=8, gru=16)
    q = rot.random_quats(10, RNG); c, p = T.encode(q, 4)
    lg = m.child_logits_tf(m.encode(torch.randn(10, 16)), torch.as_tensor(c), torch.as_tensor(p),
                           torch.as_tensor(parent_centres(c, p), dtype=torch.float32))
    assert lg.shape == (10, 4, 8)


def test_fused_beam_single_gru_matches_model_beam():
    from tfpose.fusion import fused_beam
    torch.manual_seed(2)
    m = P.HierGRU(16, 3, hidden=32, emb=8, gru=16).eval()
    x = torch.randn(25, 16)
    for beam in (1, 4):
        c1, p1, _ = m.beam_search(x, beam=beam)
        c2, p2 = fused_beam([m], [x], "single", beam=beam)
        assert torch.equal(c1[:, 0], c2) and torch.equal(p1[:, 0], p2)


def test_transformer_teacher_forcing_matches_prefix_decoding():
    torch.manual_seed(3)
    m = P.HierTransformer((16, 8), 3, d=32, heads=4, ff=64).eval()
    x = torch.randn(12, 16 * 8)
    q = rot.random_quats(12, RNG); c, p = T.encode(q, 3)
    c, p = torch.as_tensor(c), torch.as_tensor(p)
    mem = m.encode(x)
    rl, cl = m.logits_tf(mem, c, p)
    assert torch.allclose(rl, m.root(mem), atol=1e-5)
    for l in range(3):
        assert torch.allclose(cl[:, l], m.child_logits_prefix(mem, c, p[:, :l]), atol=1e-5)


def test_transformer_fused_beam_greedy():
    from tfpose.fusion import fused_beam
    torch.manual_seed(4)
    m = P.HierTransformer((16, 8), 3, d=32, heads=4, ff=64).eval()
    x = torch.randn(10, 128)
    c, p = fused_beam([m], [x], "single", beam=1)
    mem = m.encode(x); chart = m.root(mem).argmax(-1); path = torch.zeros(10, 0, dtype=torch.long)
    for l in range(3):
        path = torch.cat([path, m.child_logits_prefix(mem, chart, path).argmax(-1)[:, None]], 1)
    assert torch.equal(c, chart) and torch.equal(p, path)
    c4, p4 = fused_beam([m], [x], "single", beam=4)
    assert c4.shape == (10,) and p4.shape == (10, 3)
