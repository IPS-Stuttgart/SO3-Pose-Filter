"""Vectorized SO(3) helpers for rotation-matrix state representations."""

from __future__ import annotations

import numpy as np


EPS = 1e-8


def skew(v: np.ndarray) -> np.ndarray:
    """Return skew-symmetric matrices for vectors shaped (..., 3)."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def unskew(m: np.ndarray) -> np.ndarray:
    """Return vectors from skew-symmetric matrices shaped (..., 3, 3)."""
    m = np.asarray(m, dtype=np.float64)
    return np.stack(
        [
            0.5 * (m[..., 2, 1] - m[..., 1, 2]),
            0.5 * (m[..., 0, 2] - m[..., 2, 0]),
            0.5 * (m[..., 1, 0] - m[..., 0, 1]),
        ],
        axis=-1,
    )


def project_to_so3(r: np.ndarray) -> np.ndarray:
    """Project matrices to SO(3) with the closest orthogonal matrix."""
    r = np.asarray(r, dtype=np.float64)
    flat = r.reshape((-1, 3, 3))
    u, _, vt = np.linalg.svd(flat)
    projected = u @ vt
    det = np.linalg.det(projected)
    bad = det < 0
    if np.any(bad):
        u[bad, :, -1] *= -1.0
        projected[bad] = u[bad] @ vt[bad]
    return projected.reshape(r.shape)


def exp_map(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map from tangent vectors shaped (..., 3) to SO(3)."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec, axis=-1)
    k = skew(rotvec)
    eye = np.broadcast_to(np.eye(3), rotvec.shape[:-1] + (3, 3)).copy()

    theta2 = theta * theta
    small = theta < 1e-6
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    a[small] = 1.0 - theta2[small] / 6.0 + theta2[small] * theta2[small] / 120.0
    b[small] = 0.5 - theta2[small] / 24.0 + theta2[small] * theta2[small] / 720.0
    a[~small] = np.sin(theta[~small]) / theta[~small]
    b[~small] = (1.0 - np.cos(theta[~small])) / theta2[~small]
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def log_map(r: np.ndarray) -> np.ndarray:
    """Logarithm map from rotation matrices shaped (..., 3, 3) to rotvecs."""
    r = project_to_so3(np.asarray(r, dtype=np.float64))
    trace = np.trace(r, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    vee = unskew(r - np.swapaxes(r, -1, -2))

    out = np.zeros(r.shape[:-2] + (3,), dtype=np.float64)
    small = theta < 1e-6
    out[small] = 0.5 * vee[small]

    regular = (theta >= 1e-6) & (np.pi - theta > 1e-5)
    scale = theta[regular] / (2.0 * np.sin(theta[regular]))
    out[regular] = scale[..., None] * vee[regular]

    near_pi = ~(small | regular)
    if np.any(near_pi):
        flat_r = r[near_pi]
        flat_theta = theta[near_pi]
        vals = []
        for mat, ang in zip(flat_r, flat_theta):
            axis = np.empty(3, dtype=np.float64)
            diag = np.diag(mat)
            idx = int(np.argmax(diag))
            axis[idx] = np.sqrt(max((diag[idx] + 1.0) * 0.5, 0.0))
            denom = max(2.0 * axis[idx], EPS)
            if idx == 0:
                axis[1] = (mat[0, 1] + mat[1, 0]) / denom
                axis[2] = (mat[0, 2] + mat[2, 0]) / denom
            elif idx == 1:
                axis[0] = (mat[0, 1] + mat[1, 0]) / denom
                axis[2] = (mat[1, 2] + mat[2, 1]) / denom
            else:
                axis[0] = (mat[0, 2] + mat[2, 0]) / denom
                axis[1] = (mat[1, 2] + mat[2, 1]) / denom
            norm = np.linalg.norm(axis)
            if norm < EPS:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = axis / norm
            vals.append(axis * ang)
        out[near_pi] = np.asarray(vals)
    return out


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors shaped (..., 3) to rotation matrices."""
    return exp_map(axis_angle)


def matrix_to_axis_angle(rotations: np.ndarray) -> np.ndarray:
    """Convert rotation matrices shaped (..., 3, 3) to axis-angle vectors."""
    return log_map(rotations)


def geodesic_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return SO(3) geodesic distances between rotations, in radians."""
    rel = np.asarray(a, dtype=np.float64) @ np.swapaxes(np.asarray(b, dtype=np.float64), -1, -2)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cos_theta)


def left_apply_delta(delta: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Apply tangent-space deltas by left multiplication: exp(delta) @ R."""
    return exp_map(delta) @ rotations


def left_delta(current: np.ndarray, next_rotation: np.ndarray) -> np.ndarray:
    """Return delta satisfying approximately next_rotation = exp(delta) @ current."""
    rel = np.asarray(next_rotation, dtype=np.float64) @ np.swapaxes(
        np.asarray(current, dtype=np.float64), -1, -2
    )
    return log_map(rel)


def chordal_mean(rotations: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Compute per-joint weighted chordal means for rotations shaped (N, J, 3, 3)."""
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim != 4:
        raise ValueError(f"expected rotations shaped (N, J, 3, 3), got {rotations.shape}")
    n = rotations.shape[0]
    if weights is None:
        weights = np.full(n, 1.0 / n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    mats = np.einsum("n,njab->jab", weights, rotations)
    return project_to_so3(mats)


def mean_joint_distance_deg(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean per-joint geodesic distance in degrees."""
    dist = geodesic_distance(a, b)
    if mask is not None:
        dist = dist[np.asarray(mask, dtype=bool)]
    if dist.size == 0:
        return float("nan")
    return float(np.degrees(np.mean(dist)))
