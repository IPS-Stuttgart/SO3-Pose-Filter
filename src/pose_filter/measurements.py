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
    confidence: np.ndarray


def validate_confidence(
    confidence: np.ndarray, expected_shape: tuple[int, ...]
) -> np.ndarray:
    """Validate detector-style confidence values in [0, 1]."""
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.shape != expected_shape:
        raise ValueError(
            f"expected confidence shaped {expected_shape}, got {confidence.shape}"
        )
    if np.any(~np.isfinite(confidence)):
        raise ValueError("confidence values must be finite")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence values must be in [0, 1]")
    return confidence


def make_synthetic_measurements(
    truth: np.ndarray,
    noise_deg: float,
    occlusion_prob: float,
    rng: np.random.Generator,
    confidence_noise_std: float = 0.0,
    min_confidence: float = 0.2,
) -> SyntheticMeasurements:
    """Apply tangent Gaussian SO(3) noise, random occlusion, and confidence scores."""
    truth = np.asarray(truth, dtype=np.float64)
    sigma = np.radians(float(noise_deg))
    noise = rng.normal(0.0, sigma, size=truth.shape[:-2] + (3,))
    observations = left_apply_delta(noise, truth)
    mask = rng.random(truth.shape[:-2]) >= float(occlusion_prob)
    if mask.shape[0] > 0:
        mask[0] = True
    if confidence_noise_std < 0.0:
        raise ValueError("confidence_noise_std must be non-negative")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    confidence = mask.astype(np.float64)
    if confidence_noise_std > 0.0:
        noisy_confidence = 1.0 + rng.normal(0.0, confidence_noise_std, size=mask.shape)
        noisy_confidence = np.clip(noisy_confidence, min_confidence, 1.0)
        confidence = np.where(mask, noisy_confidence, 0.0)
    return SyntheticMeasurements(
        observations=observations,
        mask=mask,
        noise_sigma_rad=sigma,
        confidence=confidence,
    )


def log_likelihood(
    observations: np.ndarray,
    states: np.ndarray,
    mask: np.ndarray,
    noise_sigma_rad: float,
    confidence: np.ndarray | None = None,
    joint_noise_sigma_rad: np.ndarray | None = None,
) -> np.ndarray:
    """Known synthetic log-likelihood, summed over confidence-weighted joints.

    `states` can be shaped `[J, 3, 3]`, `[N, J, 3, 3]`, or `[T, J, 3, 3]`.
    The returned value has the leading state dimensions before the joint axis.
    """
    active = np.asarray(mask, dtype=bool)
    if confidence is None:
        weights = active.astype(np.float64)
    else:
        weights = np.where(active, validate_confidence(confidence, active.shape), 0.0)

    if joint_noise_sigma_rad is None:
        sigma = max(float(noise_sigma_rad), 1e-8)
    else:
        sigma = np.asarray(joint_noise_sigma_rad, dtype=np.float64)
        if sigma.shape == ():
            sigma = max(float(sigma), 1e-8)
        elif sigma.shape != active.shape:
            raise ValueError(
                f"expected joint_noise_sigma_rad shaped {active.shape}, got {sigma.shape}"
            )
        elif np.any(sigma <= 0.0) or np.any(~np.isfinite(sigma)):
            raise ValueError("joint_noise_sigma_rad values must be positive and finite")
    dist = geodesic_distance(states, observations)
    return -0.5 * np.sum(weights * (dist / sigma) ** 2, axis=-1)


def observed_error_deg(
    truth: np.ndarray,
    observations: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray | None = None,
) -> float:
    """Mean observed-joint measurement error in degrees, optionally confidence-weighted."""
    dist = geodesic_distance(truth, observations)
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return float("nan")
    if confidence is None:
        return float(np.degrees(np.mean(dist[active])))
    weights = np.where(active, validate_confidence(confidence, active.shape), 0.0)
    if np.sum(weights) <= 0.0:
        return float("nan")
    return float(np.degrees(np.sum(weights * dist) / np.sum(weights)))
