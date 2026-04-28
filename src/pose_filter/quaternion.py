"""PyRecEst-backed quaternion helpers for SO(3)^K pose states."""

from __future__ import annotations

import numpy as np
from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
    HyperhemisphereCartProdDiracDistribution,
)

from .so3 import EPS, project_to_so3

PyRecEstQuaternionDistribution = HyperhemisphereCartProdDiracDistribution


def canonicalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Normalize scalar-last quaternions and flip signs so the scalar part is nonnegative."""
    q = np.asarray(quaternions, dtype=np.float64)
    if q.shape[-1:] != (4,):
        raise ValueError(f"expected quaternions shaped (..., 4), got {q.shape}")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError("cannot normalize zero-length quaternion")
    q = q / norms
    return np.where(q[..., 3:4] < 0.0, -q, q)


def rotations_to_quaternions(rotations: np.ndarray) -> np.ndarray:
    """Convert rotation matrices shaped `(..., 3, 3)` to scalar-last quaternions."""
    raw = np.asarray(rotations, dtype=np.float64)
    if raw.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotations shaped (..., 3, 3), got {raw.shape}")
    r = project_to_so3(raw)

    flat = r.reshape((-1, 3, 3))
    q = np.empty((flat.shape[0], 4), dtype=np.float64)
    trace = flat[:, 0, 0] + flat[:, 1, 1] + flat[:, 2, 2]

    positive = trace > 0.0
    if np.any(positive):
        mats = flat[positive]
        scale = 2.0 * np.sqrt(np.maximum(trace[positive] + 1.0, EPS))
        q[positive, 3] = 0.25 * scale
        q[positive, 0] = (mats[:, 2, 1] - mats[:, 1, 2]) / scale
        q[positive, 1] = (mats[:, 0, 2] - mats[:, 2, 0]) / scale
        q[positive, 2] = (mats[:, 1, 0] - mats[:, 0, 1]) / scale

    nonpositive = ~positive
    if np.any(nonpositive):
        idxs = np.where(nonpositive)[0]
        mats = flat[idxs]
        diag = np.stack([mats[:, 0, 0], mats[:, 1, 1], mats[:, 2, 2]], axis=1)
        largest = np.argmax(diag, axis=1)

        for axis in range(3):
            selected = largest == axis
            if not np.any(selected):
                continue
            out_idx = idxs[selected]
            m = mats[selected]
            if axis == 0:
                scale = 2.0 * np.sqrt(
                    np.maximum(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], EPS)
                )
                q[out_idx, 3] = (m[:, 2, 1] - m[:, 1, 2]) / scale
                q[out_idx, 0] = 0.25 * scale
                q[out_idx, 1] = (m[:, 0, 1] + m[:, 1, 0]) / scale
                q[out_idx, 2] = (m[:, 0, 2] + m[:, 2, 0]) / scale
            elif axis == 1:
                scale = 2.0 * np.sqrt(
                    np.maximum(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], EPS)
                )
                q[out_idx, 3] = (m[:, 0, 2] - m[:, 2, 0]) / scale
                q[out_idx, 0] = (m[:, 0, 1] + m[:, 1, 0]) / scale
                q[out_idx, 1] = 0.25 * scale
                q[out_idx, 2] = (m[:, 1, 2] + m[:, 2, 1]) / scale
            else:
                scale = 2.0 * np.sqrt(
                    np.maximum(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], EPS)
                )
                q[out_idx, 3] = (m[:, 1, 0] - m[:, 0, 1]) / scale
                q[out_idx, 0] = (m[:, 0, 2] + m[:, 2, 0]) / scale
                q[out_idx, 1] = (m[:, 1, 2] + m[:, 2, 1]) / scale
                q[out_idx, 2] = 0.25 * scale

    return canonicalize_quaternions(q.reshape(r.shape[:-2] + (4,)))


def quaternions_to_rotations(quaternions: np.ndarray) -> np.ndarray:
    """Convert scalar-last quaternions shaped `(..., 4)` to rotation matrices."""
    q = canonicalize_quaternions(quaternions)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    rotations = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    rotations[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[..., 0, 1] = 2.0 * (x * y - z * w)
    rotations[..., 0, 2] = 2.0 * (x * z + y * w)
    rotations[..., 1, 0] = 2.0 * (x * y + z * w)
    rotations[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[..., 1, 2] = 2.0 * (y * z - x * w)
    rotations[..., 2, 0] = 2.0 * (x * z - y * w)
    rotations[..., 2, 1] = 2.0 * (y * z + x * w)
    rotations[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return project_to_so3(rotations)


def quaternions_to_pyrecest_hyperhemisphere_dirac(
    quaternions: np.ndarray, weights: np.ndarray | None = None
) -> HyperhemisphereCartProdDiracDistribution:
    """Create the PyRecEst backend distribution for quaternion pose states.

    Input quaternions are scalar-last and shaped `(N, J, 4)`. A single state
    shaped `(J, 4)` is accepted and treated as one Dirac component.
    """
    q = canonicalize_quaternions(quaternions)
    if q.ndim == 2:
        q = q[None, ...]
    if q.ndim != 3:
        raise ValueError(
            f"expected quaternions shaped (N, J, 4) or (J, 4), got {q.shape}"
        )

    n_particles, n_joints, _ = q.shape
    if weights is None:
        w = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n_particles,):
            raise ValueError(f"expected weights shaped {(n_particles,)}, got {w.shape}")
        total = np.sum(w)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("weights must have a positive finite sum")
        w = w / total

    return HyperhemisphereCartProdDiracDistribution(
        d=q.reshape(n_particles, n_joints * 4),
        w=w,
        dim_hemisphere=3,
        n_hemispheres=n_joints,
    )


def rotations_to_pyrecest_hyperhemisphere_dirac(
    rotations: np.ndarray, weights: np.ndarray | None = None
) -> HyperhemisphereCartProdDiracDistribution:
    """Create the PyRecEst backend distribution from SO(3)^K rotations."""
    return quaternions_to_pyrecest_hyperhemisphere_dirac(
        rotations_to_quaternions(rotations),
        weights=weights,
    )


def pyrecest_hyperhemisphere_dirac_to_quaternions(
    distribution,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(quaternions, weights)` from a PyRecEst hyperhemisphere Dirac distribution."""
    if not isinstance(distribution, HyperhemisphereCartProdDiracDistribution):
        raise TypeError("expected a PyRecEst HyperhemisphereCartProdDiracDistribution")
    if getattr(distribution, "dim_hemisphere", None) != 3:
        raise ValueError("expected a PyRecEst distribution with dim_hemisphere=3")
    n_joints = int(getattr(distribution, "n_hemispheres"))
    d = np.asarray(distribution.d, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != n_joints * 4:
        raise ValueError(
            f"expected flattened quaternions shaped (N, {n_joints * 4}), got {d.shape}"
        )
    weights = np.asarray(distribution.w, dtype=np.float64)
    return canonicalize_quaternions(d.reshape(d.shape[0], n_joints, 4)), weights.copy()


def pyrecest_hyperhemisphere_dirac_to_rotations(
    distribution,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(rotations, weights)` from a PyRecEst hyperhemisphere Dirac distribution."""
    quaternions, weights = pyrecest_hyperhemisphere_dirac_to_quaternions(distribution)
    return quaternions_to_rotations(quaternions), weights


quaternions_to_pyrecest_dirac = quaternions_to_pyrecest_hyperhemisphere_dirac
rotations_to_pyrecest_dirac = rotations_to_pyrecest_hyperhemisphere_dirac
pyrecest_dirac_to_quaternions = pyrecest_hyperhemisphere_dirac_to_quaternions
pyrecest_dirac_to_rotations = pyrecest_hyperhemisphere_dirac_to_rotations
