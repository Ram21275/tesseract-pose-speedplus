import numpy as np
import pytest

from tfpose import rotations as rot
from tfpose import tesseract as T

RNG = np.random.default_rng(0)


# ---------- rotations ----------

def test_identity_matrix():
    assert np.allclose(rot.quat_to_matrix(np.array([1.0, 0, 0, 0])), np.eye(3))


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
@pytest.mark.parametrize("angle", [np.pi / 2, np.pi, -np.pi / 3])
def test_axis_rotations_match_rodrigues(axis, angle):
    import cv2
    q = rot.axis_angle_to_quat(axis, angle)
    R_cv, _ = cv2.Rodrigues(np.array(axis, float) * angle)
    assert np.allclose(rot.quat_to_matrix(q), R_cv, atol=1e-12)


def test_180_degree_round_trip():
    for axis in np.eye(3):
        q = rot.axis_angle_to_quat(axis, np.pi)
        q2 = rot.matrix_to_quat(rot.quat_to_matrix(q))
        assert rot.geodesic_distance(q, q2) < 1e-6


def test_matrix_quat_round_trip():
    q = rot.random_quats(2000, RNG)
    R = rot.quat_to_matrix(q)
    assert np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1)
    assert np.max(rot.geodesic_distance(q, rot.matrix_to_quat(R))) < 1e-6


def test_geodesic_antipodal_and_known_angle():
    q = rot.random_quats(100, RNG)
    assert np.max(rot.geodesic_distance(q, -q)) < 1e-6
    a = rot.axis_angle_to_quat([0, 0, 1], 0.7)
    assert abs(rot.geodesic_distance(np.array([1.0, 0, 0, 0]), a) - 0.7) < 1e-12


def test_quat_multiply_matches_matrix_product():
    a, b = rot.random_quats(50, RNG), rot.random_quats(50, RNG)
    assert np.allclose(rot.quat_to_matrix(rot.quat_multiply(a, b)),
                       rot.quat_to_matrix(a) @ rot.quat_to_matrix(b), atol=1e-12)


def test_invalid_inputs_fail_loudly():
    with pytest.raises(ValueError):
        rot.normalize(np.zeros(4))
    with pytest.raises(ValueError):
        rot.normalize(np.array([np.nan, 0, 0, 1.0]))


# ---------- tesseract ----------

@pytest.mark.parametrize("depth", [1, 3, 5])
def test_antipodal_invariance(depth):
    q = rot.random_quats(20000, RNG)
    c1, p1 = T.encode(q, depth)
    c2, p2 = T.encode(-q, depth)
    assert np.array_equal(c1, c2) and np.array_equal(p1, p2)


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_encode_decode_round_trip_all_cells(depth):
    chart, path = T.all_cells(depth)
    c2, p2 = T.encode(T.decode(chart, path), depth)
    assert np.array_equal(chart, c2) and np.array_equal(path, p2)


def test_leaf_id_round_trip():
    chart, path = T.all_cells(3)
    lid = T.leaf_id(chart, path)
    assert np.array_equal(lid, np.arange(4 * 8 ** 3))
    c2, p2 = T.leaf_to_path(lid, 3)
    assert np.array_equal(chart, c2) and np.array_equal(path, p2)


def test_prefix_consistency():
    q = rot.random_quats(5000, RNG)
    _, p5 = T.encode(q, 5)
    _, p3 = T.encode(q, 3)
    assert np.array_equal(p5[:, :3], p3)


def test_finite_and_canonical():
    q = rot.random_quats(10000, RNG)
    qc, chart = T.canonicalize(q)
    assert np.all(np.isfinite(qc))
    assert np.all(qc[np.arange(len(qc)), chart] > 0)
    centres = T.codebook(3)
    assert np.all(np.isfinite(centres)) and np.allclose(np.linalg.norm(centres, axis=1), 1)


def test_tie_and_seam_determinism():
    # exact ties between |q_i| -> first index wins, and sign of -q is irrelevant
    s = 0.5
    for q in [np.array([s, s, s, s]), np.array([s, -s, s, -s]), np.array([0, 1, 1, 0.0]) / np.sqrt(2)]:
        c1, p1 = T.encode(q, 4)
        c2, p2 = T.encode(-q, 4)
        assert c1[0] == c2[0] and np.array_equal(p1, p2)
        assert c1[0] == int(np.argmax(np.abs(q)))
    # midpoint equality goes to the upper half: u = 0 -> bit 1 at level 0
    c, p = T.encode(np.array([1.0, 0, 0, 0]), 1)
    assert c[0] == 0 and p[0, 0] == 7
    # repeated calls deterministic
    q = rot.random_quats(1000, RNG)
    assert np.array_equal(T.encode(q, 5)[1], T.encode(q.copy(), 5)[1])


def test_quantization_error_decreases_with_depth():
    q = rot.random_quats(20000, RNG)
    errs = []
    for L in range(1, 6):
        c, p = T.encode(q, L)
        errs.append(np.mean(rot.geodesic_distance(q, T.decode(c, p))))
    assert all(a > b for a, b in zip(errs, errs[1:]))


def test_point_inside_own_cell_corner_bound():
    q = rot.random_quats(5000, RNG)
    c, p = T.encode(q, 3)
    centre = T.decode(c, p)
    corners = T.cell_corners(c, p)
    bound = np.max(2 * np.arccos(np.clip(np.abs(np.einsum("nd,nkd->nk", centre, corners)), 0, 1)), axis=1)
    assert np.all(rot.geodesic_distance(q, centre) <= bound + 1e-9)
