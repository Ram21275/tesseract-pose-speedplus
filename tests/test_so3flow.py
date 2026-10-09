import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfpose import so3flow as F  # noqa: E402
from tfpose import tesseract as T  # noqa: E402
from tfpose.phase3 import qmul, rotvec_to_quat  # noqa: E402

torch.manual_seed(0)


def test_exp_log_roundtrip():
    v = torch.randn(1000, 3, dtype=torch.float64); v = v / v.norm(dim=-1, keepdim=True) * torch.rand(1000, 1, dtype=torch.float64) * 3.1
    q0 = F.uniform_quats(1000, "cpu", dtype=torch.float64)
    q1 = qmul(q0, rotvec_to_quat(v))
    assert torch.allclose(F.relative_rotvec(q0, q1), v, atol=1e-8)


def test_fm_path_endpoints_and_velocity():
    q0, q1 = F.uniform_quats(500, "cpu", dtype=torch.float64), F.uniform_quats(500, "cpu", dtype=torch.float64)
    for tau, target in [(0.0, q0), (1.0, q1)]:
        qt, a, m = F.fm_pairs(q0, q1, torch.full((500,), tau, dtype=torch.float64))
        assert (F.geo_deg(qt, target)[m] < 1e-5).all()
    # velocity is constant in the body frame: finite difference along the path equals a
    t = torch.rand(500, dtype=torch.float64) * 0.9; h = 1e-6
    qa, a, m = F.fm_pairs(q0, q1, t); qb, _, _ = F.fm_pairs(q0, q1, t + h)
    assert torch.allclose(F.relative_rotvec(qa, qb)[m] / h, a[m], atol=1e-4)


def test_quat_to_mat_sign_invariant_and_orthonormal():
    q = F.uniform_quats(100, "cpu", dtype=torch.float64)
    R = F.quat_to_mat(q).view(-1, 3, 3)
    assert torch.allclose(F.quat_to_mat(-q), F.quat_to_mat(q))
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(100, 3, 3), atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(100, dtype=torch.float64))


def test_integrate_exact_for_true_velocity():
    q0, q1 = F.uniform_quats(200, "cpu", dtype=torch.float64), F.uniform_quats(200, "cpu", dtype=torch.float64)
    a = F.relative_rotvec(q0, q1)
    out = F.integrate(lambda q, tau: a, q0, nfe=7)          # constant body velocity: Euler on the group is exact
    assert (F.geo_deg(out, q1) < 1e-6).all()


def test_flow_head_shapes_and_mode():
    mem = torch.randn(6, 64, 256)
    head = F.FlowHead()
    assert head(mem, F.uniform_quats(6, "cpu"), torch.rand(6)).shape == (6, 3)
    r = F.ResidualHead040()
    assert torch.equal(r(mem, F.uniform_quats(6, "cpu")), torch.zeros(6, 3))   # zero-init
    s = F.uniform_quats(4 * 9, "cpu").view(4, 9, 4)
    s[:, :5] = s[:, :1]                                                            # 5 identical samples form the mode
    assert torch.allclose(F.kde_mode(s), s[:, 0])


def test_same_samples_give_exact_parent_child_masses():
    q = F.uniform_quats(5000, "cpu").numpy()
    chart, path = T.encode(q, 5)
    for l in range(1, 5):
        parent = T.leaf_id(chart, path[:, :l]) if l else chart
        child = T.leaf_id(chart, path[:, :l + 1])
        # the mass of each parent equals the summed masses of its children
        pm = {k: v for k, v in zip(*np.unique(parent, return_counts=True))}
        cm = {}
        for p_, c_ in zip(parent, child):
            cm.setdefault(p_, set()).add(c_)
        cc = dict(zip(*np.unique(child, return_counts=True)))
        assert all(pm[p_] == sum(cc[c_] for c_ in cs) for p_, cs in cm.items())
