"""Synthetic SO(3)^K measurement generation and likelihoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .so3 import geodesic_distance, left_apply_delta


@dataclass(frozen=True)
class SyntheticMeasurements:
    observations: np.ndarray
    mask: np.ndarray
    noise_sigma_rad: float


def make_synthetic_measurements(
    truth: np.ndarray,
    noise_deg: float,
    occlusion_prob: float,
    rng: np.random.Generator,
) -> SyntheticMeasurements:
    """Apply tangent Gaussian SO(3) noise and random per-joint occlusion."""
    truth = np.asarray(truth, dtype=np.float64)
    sigma = np.radians(float(noise_deg))
    noise = rng.normal(0.0, sigma, size=truth.shape[:-2] + (3,))
    observations = left_apply_delta(noise, truth)
    mask = rng.random(truth.shape[:-2]) >= float(occlusion_prob)
    if mask.shape[0] > 0:
        mask[0] = True
    return SyntheticMeasurements(
        observations=observations, mask=mask, noise_sigma_rad=sigma
    )


def log_likelihood(
    observations: np.ndarray,
    states: np.ndarray,
    mask: np.ndarray,
    noise_sigma_rad: float,
) -> np.ndarray:
    """Known synthetic log-likelihood, summed over observed joints.

    `states` can be shaped `[J, 3, 3]`, `[N, J, 3, 3]`, or `[T, J, 3, 3]`.
    The returned value has the leading state dimensions before the joint axis.
    """
    sigma = max(float(noise_sigma_rad), 1e-8)
    dist = geodesic_distance(states, observations)
    active = np.asarray(mask, dtype=bool)
    return -0.5 * np.sum(np.where(active, (dist / sigma) ** 2, 0.0), axis=-1)


def observed_error_deg(
    truth: np.ndarray, observations: np.ndarray, mask: np.ndarray
) -> float:
    """Mean observed-joint measurement error in degrees."""
    dist = geodesic_distance(truth, observations)
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return float("nan")
    return float(np.degrees(np.mean(dist[active])))
