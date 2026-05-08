"""PyRecEst-backed quaternion helpers for SO(3)^K pose states."""

from __future__ import annotations

from typing import Any

import numpy as np
from pyrecest.distributions import (  # type: ignore[import-untyped]
    ConversionError,
    SO3DiracDistribution,
    SO3ProductDiracDistribution,
    convert_distribution,
)
from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (  # type: ignore[import-untyped]
    HyperhemisphereCartProdDiracDistribution,
)

from .so3 import project_to_so3

PyRecEstQuaternionDistribution = SO3ProductDiracDistribution


def _as_numpy(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def canonicalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Normalize scalar-last quaternions and flip signs so the scalar part is nonnegative."""
    q = np.asarray(quaternions, dtype=np.float64)
    if q.shape[-1:] != (4,):
        raise ValueError(f"expected quaternions shaped (..., 4), got {q.shape}")

    try:
        canonical = SO3DiracDistribution(q.reshape((-1, 4))).as_quaternions()
    except AssertionError as exc:
        raise ValueError(str(exc)) from exc

    return _as_numpy(canonical).reshape(q.shape)


def _as_product_rotation_particles(rotations: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return rotations as PyRecEst SO(3)^K particles plus the original product shape."""
    raw = np.asarray(rotations, dtype=np.float64)
    if raw.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotations shaped (..., 3, 3), got {raw.shape}")

    projected = project_to_so3(raw)
    if projected.ndim == 2:
        return projected.reshape((1, 1, 3, 3)), ()

    product_shape = tuple(projected.shape[:-2])
    num_rotations = int(product_shape[-1])
    particle_shape = product_shape[:-1]
    num_particles = int(np.prod(particle_shape)) if particle_shape else 1
    return projected.reshape((num_particles, num_rotations, 3, 3)), product_shape


def rotations_to_quaternions(rotations: np.ndarray) -> np.ndarray:
    """Convert rotation matrices shaped `(..., 3, 3)` to scalar-last quaternions."""
    product_particles, output_shape = _as_product_rotation_particles(rotations)
    from_rotation_matrices = getattr(SO3ProductDiracDistribution, "from_rotation_matrices", None)
    if not callable(from_rotation_matrices):
        raise ImportError(
            "rotations_to_quaternions requires PyRecEst with "
            "SO3ProductDiracDistribution.from_rotation_matrices available"
        )

    distribution = from_rotation_matrices(product_particles)
    quaternions = _as_numpy(distribution.as_quaternions()).reshape(output_shape + (4,))
    return canonicalize_quaternions(quaternions)


def quaternions_to_rotations(quaternions: np.ndarray) -> np.ndarray:
    """Convert scalar-last quaternions shaped `(..., 4)` to rotation matrices."""
    q = canonicalize_quaternions(quaternions)
    rotations = SO3ProductDiracDistribution.as_rotation_matrices(q)
    return project_to_so3(_as_numpy(rotations))


def _normalize_weights(weights: np.ndarray | None, n_particles: int) -> np.ndarray | None:
    if weights is None:
        return None
    normalized = np.asarray(weights, dtype=np.float64)
    if normalized.shape != (n_particles,):
        raise ValueError(f"expected weights shaped {(n_particles,)}, got {normalized.shape}")
    total = np.sum(normalized)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have a positive finite sum")
    return normalized / total


def quaternions_to_pyrecest_hyperhemisphere_dirac(quaternions: np.ndarray, weights: np.ndarray | None = None) -> SO3ProductDiracDistribution:
    """Create the PyRecEst SO(3)^K Dirac distribution for quaternion pose states.

    Input quaternions are scalar-last and shaped `(N, J, 4)`. A single state
    shaped `(J, 4)` is accepted and treated as one Dirac component.
    """
    q = canonicalize_quaternions(quaternions)
    if q.ndim == 2:
        n_particles = 1
        num_rotations = q.shape[0]
    elif q.ndim == 3:
        n_particles = q.shape[0]
        num_rotations = None
    else:
        raise ValueError(f"expected quaternions shaped (N, J, 4) or (J, 4), got {q.shape}")

    return SO3ProductDiracDistribution(
        q,
        w=_normalize_weights(weights, n_particles),
        num_rotations=num_rotations,
    )


def rotations_to_pyrecest_hyperhemisphere_dirac(rotations: np.ndarray, weights: np.ndarray | None = None) -> SO3ProductDiracDistribution:
    """Create the PyRecEst SO(3)^K Dirac distribution from SO(3)^K rotations."""
    return quaternions_to_pyrecest_hyperhemisphere_dirac(
        rotations_to_quaternions(rotations),
        weights=weights,
    )


def _as_so3_product_dirac(distribution: Any) -> SO3ProductDiracDistribution | None:
    if isinstance(distribution, SO3ProductDiracDistribution):
        return distribution
    try:
        converted = convert_distribution(distribution, "so3_product_dirac")
    except (ConversionError, TypeError, ValueError, AssertionError):
        return None
    if isinstance(converted, SO3ProductDiracDistribution):
        return converted
    return None


def pyrecest_hyperhemisphere_dirac_to_quaternions(
    distribution,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(quaternions, weights)` from a PyRecEst SO(3)^K Dirac distribution."""
    product_distribution = _as_so3_product_dirac(distribution)
    if product_distribution is not None:
        return (
            canonicalize_quaternions(_as_numpy(product_distribution.as_quaternions())),
            _as_numpy(product_distribution.w).copy(),
        )

    if not isinstance(distribution, HyperhemisphereCartProdDiracDistribution):
        raise TypeError("expected a PyRecEst SO3ProductDiracDistribution or HyperhemisphereCartProdDiracDistribution")
    if getattr(distribution, "dim_hemisphere", None) != 3:
        raise ValueError("expected a PyRecEst distribution with dim_hemisphere=3")
    n_joints = int(getattr(distribution, "n_hemispheres"))
    d = _as_numpy(distribution.d)
    if d.ndim == 3 and d.shape[1:] == (n_joints, 4):
        quaternions = d
    elif d.ndim == 2 and d.shape[1] == n_joints * 4:
        quaternions = d.reshape(d.shape[0], n_joints, 4)
    else:
        raise ValueError(f"expected quaternions shaped (N, {n_joints}, 4) or (N, {n_joints * 4}), got {d.shape}")
    weights = _as_numpy(distribution.w)
    return canonicalize_quaternions(quaternions), weights.copy()


def pyrecest_hyperhemisphere_dirac_to_rotations(
    distribution,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(rotations, weights)` from a PyRecEst SO(3)^K Dirac distribution."""
    quaternions, weights = pyrecest_hyperhemisphere_dirac_to_quaternions(distribution)
    return quaternions_to_rotations(quaternions), weights


quaternions_to_pyrecest_dirac = quaternions_to_pyrecest_hyperhemisphere_dirac
rotations_to_pyrecest_dirac = rotations_to_pyrecest_hyperhemisphere_dirac
pyrecest_dirac_to_quaternions = pyrecest_hyperhemisphere_dirac_to_quaternions
pyrecest_dirac_to_rotations = pyrecest_hyperhemisphere_dirac_to_rotations
