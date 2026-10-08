import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tfpose import cass


def _reference(Kt, Ks, gt, gs, scale):
    """Straightforward per-head CASS fusion (float64, full eigendecompositions)."""
    H, N, d = Kt.shape
    ks = Ks.permute(0, 2, 1).reshape(H, d, gs, gs)
    ks = F.interpolate(ks.float(), size=(gt, gt), mode="bilinear", align_corners=False).reshape(H, d, N).permute(0, 2, 1).double()
    kt = Kt.double()
    ks = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12) * (kt.max() - kt.min()) + kt.min()
    At = [kt[h] @ kt[h].T * scale for h in range(H)]
    As = [ks[h] @ ks[h].T * scale for h in range(H)]
    spec = lambda A: torch.sort(torch.linalg.eigvalsh(A), descending=True)[0][:cass.K_EIGEN]
    st = torch.stack([spec(a) for a in At]); ss = torch.stack([spec(a) for a in As])
    pa = torch.sort(F.normalize(F.normalize(ss, p=2, dim=1), p=1, dim=1), dim=1)[0]
    pb = torch.sort(F.normalize(F.normalize(st, p=2, dim=1), p=1, dim=1), dim=1)[0]
    cost = (1 - (pa[:, None] - pb[None]).abs().sum(-1)).numpy()
    rows, cols = linear_sum_assignment(cost)
    yy, xx = torch.meshgrid(torch.arange(gt), torch.arange(gt), indexing="ij")
    p = torch.stack([yy.reshape(-1), xx.reshape(-1)], 1).double()
    prior = torch.exp(-torch.cdist(p, p) ** 2 / (2 * cass.GAUSS_STD ** 2))
    out = torch.empty(H, N, N, dtype=torch.float64)
    for i, j in zip(rows, cols):
        w = 1 - cost[i, j]
        e, U = torch.linalg.eigh(As[i]); e, U = e.flip(0), U.flip(1)
        cum = torch.cumsum(e, 0) / e.sum()
        k = int((cum < cass.ENERGY).sum()) + 1
        ek = e[:k]; smin, smax = ek.min(), ek.max()
        ek = (ek - smin) / (smax - smin) * (cass.EPSILON * smax - (2 - cass.EPSILON) * smin) + (2 - cass.EPSILON) * smin
        A = U[:, :k] @ torch.diag(ek) @ U[:, :k].T
        A.fill_diagonal_(0)
        A = (A - A.min()) / (A.max() - A.min()) * (At[j].max() - At[j].min()) + At[j].min()
        out[j] = torch.softmax((w * cass.SCALE_FACTOR * A + At[j]) / (w * cass.SCALE_FACTOR + 1) + prior, -1)
    return out, rows, cols


def test_vectorized_fusion_matches_reference():
    torch.manual_seed(0)
    H, d = 4, 8
    # low-rank-ish keys so the 0.95-energy rank is < d
    Kt = (torch.randn(H, 36, 3) @ torch.randn(H, 3, d)) + 0.05 * torch.randn(H, 36, d)
    Ks = (torch.randn(H, 16, 3) @ torch.randn(H, 3, d)) + 0.05 * torch.randn(H, 16, d)
    got, info = cass.fuse_attention(Kt, Ks, 6, 4, d ** -0.5)
    ref, rows, cols = _reference(Kt, Ks, 6, 4, d ** -0.5)
    assert np.array_equal(info["pairs"][:, 0], rows) and np.array_equal(info["pairs"][:, 1], cols)
    assert torch.allclose(got.double(), ref, atol=2e-4), float((got.double() - ref).abs().max())
    assert torch.allclose(got.sum(-1), torch.ones(H, 36), atol=1e-5)
