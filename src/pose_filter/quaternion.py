"""Quaternion and PyRecEst bridge helpers for SO(3)^K pose states."""

from __future__ import annotations

import numpy as np
import quaternion

from .so3 import EPS, project_to_so3


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


def _from_numpy_quaternion_order(quaternions: np.ndarray) -> np.ndarray:
    """Convert scalar-first `(w, x, y, z)` arrays to scalar-last `(x, y, z, w)`."""
    return np.concatenate([quaternions[..., 1:], quaternions[..., :1]], axis=-1)


def _to_numpy_quaternion_order(quaternions: np.ndarray) -> np.ndarray:
    """Convert scalar-last `(x, y, z, w)` arrays to scalar-first `(w, x, y, z)`."""
    return np.concatenate([quaternions[..., 3:], quaternions[..., :3]], axis=-1)


def rotations_to_quaternions(rotations: np.ndarray) -> np.ndarray:
    """Convert rotation matrices shaped `(..., 3, 3)` to scalar-last quaternions."""
    raw = np.asarray(rotations, dtype=np.float64)
    if raw.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotations shaped (..., 3, 3), got {raw.shape}")
    r = project_to_so3(raw)
    q = quaternion.as_float_array(
        quaternion.from_rotation_matrix(r, nonorthogonal=False)
    )
    return canonicalize_quaternions(_from_numpy_quaternion_order(q))


def quaternions_to_rotations(quaternions: np.ndarray) -> np.ndarray:
    """Convert scalar-last quaternions shaped `(..., 4)` to rotation matrices."""
    q = canonicalize_quaternions(quaternions)
    q = quaternion.from_float_array(_to_numpy_quaternion_order(q))
    rotations = quaternion.as_rotation_matrix(q)
    return project_to_so3(rotations)


def quaternions_to_pyrecest_hyperhemisphere_dirac(
    quaternions: np.ndarray, weights: np.ndarray | None = None
):
    """Create a PyRecEst `HyperhemisphereCartProdDiracDistribution`.

    Input quaternions are scalar-last and shaped `(N, J, 4)`. A single state
    shaped `(J, 4)` is accepted and treated as one Dirac component.
    """
    q = canonicalize_quaternions(quaternions)
    if q.ndim == 2:
        q = q[None, ...]
    if q.ndim != 3:
        raise ValueError(f"expected quaternions shaped (N, J, 4) or (J, 4), got {q.shape}")

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

    from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
        HyperhemisphereCartProdDiracDistribution,
    )

    return HyperhemisphereCartProdDiracDistribution(
        d=q.reshape(n_particles, n_joints * 4),
        w=w,
        dim_hemisphere=3,
        n_hemispheres=n_joints,
    )


def rotations_to_pyrecest_hyperhemisphere_dirac(
    rotations: np.ndarray, weights: np.ndarray | None = None
):
    """Create a PyRecEst hyperhemisphere Dirac distribution from SO(3)^K rotations."""
    return quaternions_to_pyrecest_hyperhemisphere_dirac(
        rotations_to_quaternions(rotations),
        weights=weights,
    )


def pyrecest_hyperhemisphere_dirac_to_quaternions(distribution) -> tuple[np.ndarray, np.ndarray]:
    """Return `(quaternions, weights)` from a PyRecEst hyperhemisphere Dirac distribution."""
    if getattr(distribution, "dim_hemisphere", None) != 3:
        raise ValueError("expected a PyRecEst distribution with dim_hemisphere=3")
    n_joints = int(getattr(distribution, "n_hemispheres"))
    d = np.asarray(distribution.d, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != n_joints * 4:
        raise ValueError(f"expected flattened quaternions shaped (N, {n_joints * 4}), got {d.shape}")
    weights = np.asarray(distribution.w, dtype=np.float64)
    return canonicalize_quaternions(d.reshape(d.shape[0], n_joints, 4)), weights.copy()


def pyrecest_hyperhemisphere_dirac_to_rotations(distribution) -> tuple[np.ndarray, np.ndarray]:
    """Return `(rotations, weights)` from a PyRecEst hyperhemisphere Dirac distribution."""
    quaternions, weights = pyrecest_hyperhemisphere_dirac_to_quaternions(distribution)
    return quaternions_to_rotations(quaternions), weights


quaternions_to_pyrecest_dirac = quaternions_to_pyrecest_hyperhemisphere_dirac
rotations_to_pyrecest_dirac = rotations_to_pyrecest_hyperhemisphere_dirac
pyrecest_dirac_to_quaternions = pyrecest_hyperhemisphere_dirac_to_quaternions
pyrecest_dirac_to_rotations = pyrecest_hyperhemisphere_dirac_to_rotations
