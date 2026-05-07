"""Synthetic and detector-like SO(3)^K measurement generation and likelihoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .so3 import geodesic_distance, left_apply_delta, project_to_so3


@dataclass(frozen=True)
class SyntheticMeasurements:
    observations: np.ndarray
    mask: np.ndarray
    noise_sigma_rad: float
    confidence: np.ndarray
    joint_noise_sigma_rad: np.ndarray | None = None
    outlier_mask: np.ndarray | None = None
    occlusion_model: str = "iid"
    outlier_prob: float = 0.0


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return value


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


def confidence_to_noise_sigma(
    confidence: np.ndarray,
    sigma_min_rad: float,
    sigma_max_rad: float,
    gamma: float = 1.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map confidence to per-frame, per-joint SO(3) noise scales.

    The mapping is
    ``sigma^2(c) = sigma_min^2 + (1 - c)^gamma * (sigma_max^2 - sigma_min^2)``.
    Occluded joints receive ``sigma_max`` when ``mask`` is supplied; they still
    contribute zero likelihood because the mask deactivates them.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    if np.any(~np.isfinite(confidence)):
        raise ValueError("confidence values must be finite")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence values must be in [0, 1]")
    sigma_min = float(sigma_min_rad)
    sigma_max = float(sigma_max_rad)
    gamma = float(gamma)
    if not np.isfinite(sigma_min) or sigma_min <= 0.0:
        raise ValueError("sigma_min_rad must be positive and finite")
    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError("sigma_max_rad must be positive and finite")
    if sigma_max < sigma_min:
        raise ValueError("sigma_max_rad must be greater than or equal to sigma_min_rad")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")

    variance = sigma_min * sigma_min + np.power(1.0 - confidence, gamma) * (
        sigma_max * sigma_max - sigma_min * sigma_min
    )
    sigma = np.sqrt(np.maximum(variance, 1e-16))
    if mask is not None:
        active = np.asarray(mask, dtype=bool)
        if active.shape != confidence.shape:
            raise ValueError(f"expected mask shaped {confidence.shape}, got {active.shape}")
        sigma = np.where(active, sigma, sigma_max)
    return sigma


def make_occlusion_mask(
    shape: tuple[int, ...],
    occlusion_prob: float,
    rng: np.random.Generator,
    occlusion_model: str = "iid",
    occlusion_entry_prob: float | None = None,
    occlusion_recovery_prob: float | None = None,
    force_first_frame_visible: bool = True,
) -> np.ndarray:
    """Generate observed-joint masks with IID or bursty Markov occlusion.

    ``occlusion_model='iid'`` preserves independent Bernoulli dropout.
    ``occlusion_model='markov'`` uses a two-state visible/hidden Markov chain per
    joint. ``occlusion_entry_prob`` is P(visible -> hidden), while
    ``occlusion_recovery_prob`` is P(hidden -> visible). If the entry probability
    is omitted, it is selected to approximately match ``occlusion_prob`` as the
    stationary hidden probability for the chosen recovery probability.
    """
    if len(shape) < 1:
        raise ValueError("measurement shape must include a frame axis")
    t_steps = int(shape[0])
    occlusion_prob = _validate_probability("occlusion_prob", occlusion_prob)
    model = str(occlusion_model).lower()
    if model in {"iid", "independent", "bernoulli"}:
        mask = rng.random(shape) >= occlusion_prob
        if force_first_frame_visible and t_steps > 0:
            mask[0] = True
        return mask
    if model not in {"markov", "bursty", "temporal"}:
        raise ValueError("occlusion_model must be 'iid' or 'markov'")

    if occlusion_recovery_prob is None:
        if occlusion_prob <= 0.0:
            recovery_prob = 1.0
        elif occlusion_prob >= 1.0:
            recovery_prob = 0.0
        else:
            max_recovery = (1.0 - occlusion_prob) / occlusion_prob
            recovery_prob = min(0.25, max_recovery)
    else:
        recovery_prob = _validate_probability(
            "occlusion_recovery_prob", occlusion_recovery_prob
        )

    if occlusion_entry_prob is None:
        if occlusion_prob <= 0.0:
            entry_prob = 0.0
        elif occlusion_prob >= 1.0:
            entry_prob = 1.0
        else:
            entry_prob = occlusion_prob * recovery_prob / (1.0 - occlusion_prob)
            entry_prob = float(np.clip(entry_prob, 0.0, 1.0))
    else:
        entry_prob = _validate_probability("occlusion_entry_prob", occlusion_entry_prob)

    mask = np.empty(shape, dtype=bool)
    if t_steps == 0:
        return mask
    mask[0] = rng.random(shape[1:]) >= occlusion_prob
    if force_first_frame_visible:
        mask[0] = True
    for t in range(1, t_steps):
        was_visible = mask[t - 1]
        stay_visible = rng.random(shape[1:]) >= entry_prob
        recover = rng.random(shape[1:]) < recovery_prob
        mask[t] = np.where(was_visible, stay_visible, recover)
    return mask


def random_rotations(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Sample Haar-uniform SO(3) rotations with scalar-last unit quaternions."""
    u1 = rng.random(shape)
    u2 = rng.random(shape)
    u3 = rng.random(shape)
    two_pi = 2.0 * np.pi
    q = np.empty(shape + (4,), dtype=np.float64)
    q[..., 0] = np.sqrt(1.0 - u1) * np.sin(two_pi * u2)
    q[..., 1] = np.sqrt(1.0 - u1) * np.cos(two_pi * u2)
    q[..., 2] = np.sqrt(u1) * np.sin(two_pi * u3)
    q[..., 3] = np.sqrt(u1) * np.cos(two_pi * u3)

    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]
    rotations = np.empty(shape + (3, 3), dtype=np.float64)
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


def make_synthetic_measurements(
    truth: np.ndarray,
    noise_deg: float,
    occlusion_prob: float,
    rng: np.random.Generator,
    confidence_noise_std: float = 0.0,
    min_confidence: float = 0.2,
    occlusion_model: str = "iid",
    occlusion_entry_prob: float | None = None,
    occlusion_recovery_prob: float | None = None,
    outlier_prob: float = 0.0,
    outlier_noise_deg: float | None = None,
    outlier_mode: str = "uniform",
    confidence_calibrated_noise: bool = False,
    confidence_noise_min_deg: float | None = None,
    confidence_noise_max_deg: float | None = None,
    confidence_noise_gamma: float = 1.0,
) -> SyntheticMeasurements:
    """Apply SO(3) noise, occlusion, confidence scores, and optional outliers.

    Defaults preserve the original controlled model: independent dropout,
    Gaussian tangent noise with scalar ``noise_deg``, no outliers, and confidence
    equal to one for visible joints. More realistic corruption is enabled by
    ``occlusion_model='markov'``, ``outlier_prob > 0``, and/or
    ``confidence_calibrated_noise=True``.
    """
    truth = np.asarray(truth, dtype=np.float64)
    sigma = np.radians(float(noise_deg))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("noise_deg must be positive and finite")
    if confidence_noise_std < 0.0:
        raise ValueError("confidence_noise_std must be non-negative")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    outlier_prob = _validate_probability("outlier_prob", outlier_prob)

    measurement_shape = truth.shape[:-2]
    mask = make_occlusion_mask(
        measurement_shape,
        occlusion_prob,
        rng,
        occlusion_model=occlusion_model,
        occlusion_entry_prob=occlusion_entry_prob,
        occlusion_recovery_prob=occlusion_recovery_prob,
        force_first_frame_visible=True,
    )

    confidence = mask.astype(np.float64)
    if confidence_noise_std > 0.0:
        noisy_confidence = 1.0 + rng.normal(0.0, confidence_noise_std, size=mask.shape)
        noisy_confidence = np.clip(noisy_confidence, min_confidence, 1.0)
        confidence = np.where(mask, noisy_confidence, 0.0)

    if confidence_calibrated_noise:
        sigma_min = np.radians(
            float(confidence_noise_min_deg if confidence_noise_min_deg is not None else noise_deg)
        )
        sigma_max = np.radians(
            float(
                confidence_noise_max_deg
                if confidence_noise_max_deg is not None
                else max(noise_deg, 3.0 * noise_deg)
            )
        )
        joint_sigma = confidence_to_noise_sigma(
            confidence,
            sigma_min,
            sigma_max,
            gamma=confidence_noise_gamma,
            mask=mask,
        )
        noise = rng.normal(0.0, joint_sigma[..., None], size=measurement_shape + (3,))
    else:
        joint_sigma = None
        noise = rng.normal(0.0, sigma, size=measurement_shape + (3,))
    observations = left_apply_delta(noise, truth)

    outlier_mask = np.zeros(measurement_shape, dtype=bool)
    if outlier_prob > 0.0:
        outlier_mask = (rng.random(measurement_shape) < outlier_prob) & mask
        mode = str(outlier_mode).lower()
        if mode == "uniform":
            replacement = random_rotations(measurement_shape, rng)
        elif mode in {"large_noise", "heavy_noise"}:
            outlier_sigma = np.radians(
                float(outlier_noise_deg if outlier_noise_deg is not None else max(45.0, 6.0 * noise_deg))
            )
            replacement = left_apply_delta(
                rng.normal(0.0, outlier_sigma, size=measurement_shape + (3,)), truth
            )
        else:
            raise ValueError("outlier_mode must be 'uniform' or 'large_noise'")
        observations = np.where(outlier_mask[..., None, None], replacement, observations)

    return SyntheticMeasurements(
        observations=observations,
        mask=mask,
        noise_sigma_rad=sigma,
        confidence=confidence,
        joint_noise_sigma_rad=joint_sigma,
        outlier_mask=outlier_mask,
        occlusion_model=str(occlusion_model).lower(),
        outlier_prob=outlier_prob,
    )


def _prepare_joint_sigma(
    active_shape: tuple[int, ...],
    noise_sigma_rad: float,
    joint_noise_sigma_rad: np.ndarray | None,
) -> np.ndarray | float:
    if joint_noise_sigma_rad is None:
        return max(float(noise_sigma_rad), 1e-8)
    sigma = np.asarray(joint_noise_sigma_rad, dtype=np.float64)
    if sigma.shape == ():
        return max(float(sigma), 1e-8)
    if sigma.shape != active_shape:
        raise ValueError(
            f"expected joint_noise_sigma_rad shaped {active_shape}, got {sigma.shape}"
        )
    if np.any(sigma <= 0.0) or np.any(~np.isfinite(sigma)):
        raise ValueError("joint_noise_sigma_rad values must be positive and finite")
    return np.maximum(sigma, 1e-8)


def joint_log_likelihood(
    observations: np.ndarray,
    states: np.ndarray,
    mask: np.ndarray,
    noise_sigma_rad: float,
    confidence: np.ndarray | None = None,
    joint_noise_sigma_rad: np.ndarray | None = None,
    outlier_prob: float = 0.0,
) -> np.ndarray:
    """Per-joint log-likelihood with optional confidence and outlier mixture.

    Returns one value per joint and per leading state dimension. Summing over the
    last axis gives the sequence-frame likelihood used by the particle filter.
    The outlier component is an unnormalized constant floor, adequate for robust
    particle weights and intentionally compatible with the previous likelihood.
    """
    active = np.asarray(mask, dtype=bool)
    if confidence is None:
        weights = active.astype(np.float64)
    else:
        weights = np.where(active, validate_confidence(confidence, active.shape), 0.0)
    sigma = _prepare_joint_sigma(active.shape, noise_sigma_rad, joint_noise_sigma_rad)
    eps = _validate_probability("outlier_prob", outlier_prob)

    dist = geodesic_distance(states, observations)
    clean_ll = -0.5 * (dist / sigma) ** 2
    if eps > 0.0:
        clean_term = np.log1p(-min(eps, 1.0 - 1e-12)) + clean_ll
        outlier_term = np.log(max(eps, 1e-12))
        clean_ll = np.logaddexp(clean_term, outlier_term)
    return weights * clean_ll


def log_likelihood(
    observations: np.ndarray,
    states: np.ndarray,
    mask: np.ndarray,
    noise_sigma_rad: float,
    confidence: np.ndarray | None = None,
    joint_noise_sigma_rad: np.ndarray | None = None,
    outlier_prob: float = 0.0,
) -> np.ndarray:
    """Known synthetic log-likelihood, summed over active joints.

    `states` can be shaped `[J, 3, 3]`, `[N, J, 3, 3]`, or `[T, J, 3, 3]`.
    The returned value has the leading state dimensions before the joint axis.
    """
    return np.sum(
        joint_log_likelihood(
            observations,
            states,
            mask,
            noise_sigma_rad,
            confidence=confidence,
            joint_noise_sigma_rad=joint_noise_sigma_rad,
            outlier_prob=outlier_prob,
        ),
        axis=-1,
    )


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
