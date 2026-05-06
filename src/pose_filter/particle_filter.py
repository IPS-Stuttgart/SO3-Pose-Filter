"""Particle filter on product rotation states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .so3 import chordal_mean, geodesic_distance, left_apply_delta, left_delta
from .transitions import TransitionModel


@dataclass
class ParticleFilterResult:
    estimates: np.ndarray
    effective_sample_size: np.ndarray
    resampled: np.ndarray
    particle_spread_deg: np.ndarray


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling indices for normalized weights."""
    n = weights.shape[0]
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def _normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    return weights, np.log(weights + 1e-300)


def _normalize_log_weights_axis0(
    log_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shifted = log_weights - np.max(log_weights, axis=0, keepdims=True)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights, axis=0, keepdims=True)
    return weights, np.log(weights + 1e-300)


def _prepare_confidence(mask: np.ndarray, confidence: np.ndarray | None) -> np.ndarray:
    if confidence is None:
        return mask.astype(np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.shape != mask.shape:
        raise ValueError(
            f"expected confidence shaped {mask.shape}, got {confidence.shape}"
        )
    if np.any(~np.isfinite(confidence)):
        raise ValueError("confidence values must be finite")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence values must be in [0, 1]")
    return np.where(mask, confidence, 0.0)


def _prepare_joint_noise(
    noise_sigma_rad: float,
    mask: np.ndarray,
    joint_noise_sigma_rad: np.ndarray | None,
) -> np.ndarray:
    if joint_noise_sigma_rad is None:
        return np.full(mask.shape, max(float(noise_sigma_rad), 1e-8), dtype=np.float64)
    sigma = np.asarray(joint_noise_sigma_rad, dtype=np.float64)
    if sigma.shape == ():
        sigma = np.full(mask.shape, float(sigma), dtype=np.float64)
    elif sigma.shape != mask.shape:
        raise ValueError(
            f"expected joint_noise_sigma_rad shaped {mask.shape}, got {sigma.shape}"
        )
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("joint_noise_sigma_rad values must be positive and finite")
    return np.maximum(sigma, 1e-8)


def _particle_spread_deg(particles: np.ndarray, estimate: np.ndarray) -> float:
    dist = geodesic_distance(particles, np.asarray(estimate, dtype=np.float64)[None, ...])
    return float(np.degrees(np.mean(dist)))


def initialize_particles(
    first_observation: np.ndarray,
    first_mask: np.ndarray,
    num_particles: int,
    noise_sigma_rad: float,
    rng: np.random.Generator,
    jitter_scale: float = 0.25,
) -> np.ndarray:
    """Initialize around the first observation, with identity for initially hidden joints."""
    obs = np.asarray(first_observation, dtype=np.float64).copy()
    mask = np.asarray(first_mask, dtype=bool)
    identity = np.broadcast_to(np.eye(3), obs.shape)
    obs = np.where(mask[..., None, None], obs, identity)
    base = np.repeat(obs[None, ...], int(num_particles), axis=0)
    jitter = rng.normal(
        0.0,
        max(noise_sigma_rad * float(jitter_scale), 1e-6),
        size=base.shape[:-2] + (3,),
    )
    return left_apply_delta(jitter, base)


def run_particle_filter(
    observations: np.ndarray,
    mask: np.ndarray,
    transition_model: TransitionModel,
    noise_sigma_rad: float,
    num_particles: int,
    rng: np.random.Generator,
    resample_threshold: float = 0.5,
    factorized_update: bool = True,
    proposal_gain: float = 0.2,
    confidence: np.ndarray | None = None,
    joint_noise_sigma_rad: np.ndarray | None = None,
) -> ParticleFilterResult:
    """Run a guided bootstrap particle filter on one sequence.

    `proposal_gain` applies a small SO(3) correction toward observed joints before
    weighting. This keeps the low-particle prototype useful in the high-dimensional
    product space without changing the measurement likelihood used for scoring.
    `confidence` is shaped `[T, J]`; zero confidence is equivalent to an occluded
    joint, while values between zero and one downweight proposal correction and
    measurement likelihood. `joint_noise_sigma_rad` can override the scalar
    measurement noise with per-frame, per-joint standard deviations.
    """
    observations = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    confidence = _prepare_confidence(mask, confidence)
    joint_noise_sigma_rad = _prepare_joint_noise(
        noise_sigma_rad, mask, joint_noise_sigma_rad
    )
    t_steps = observations.shape[0]
    particles = initialize_particles(
        observations[0], confidence[0] > 0.0, num_particles, noise_sigma_rad, rng
    )
    log_weights = np.full(num_particles, -np.log(num_particles), dtype=np.float64)
    log_joint_weights = np.full(
        (num_particles, observations.shape[1]), -np.log(num_particles), dtype=np.float64
    )
    estimates = []
    ess_values = []
    resampled_flags = []
    spread_values = []
    particle_history: list[np.ndarray] = []
    history_keep = int(getattr(transition_model, "history_length", 0)) + 1

    for t in range(t_steps):
        if t > 0:
            particles = transition_model.sample_next_from_history(
                particle_history or [particles], rng
            )

        if proposal_gain > 0.0:
            delta_to_observation = left_delta(particles, observations[t])
            correction_weight = float(proposal_gain) * confidence[t][None, :, None]
            correction = correction_weight * delta_to_observation
            particles = left_apply_delta(correction, particles)

        dist = geodesic_distance(particles, observations[t])
        joint_sigma = joint_noise_sigma_rad[t][None, :]
        joint_ll = -0.5 * confidence[t][None, :] * (dist / joint_sigma) ** 2

        if factorized_update:
            joint_weights, log_joint_weights = _normalize_log_weights_axis0(
                log_joint_weights + joint_ll
            )
            estimate = []
            for joint_idx in range(observations.shape[1]):
                estimate.append(
                    chordal_mean(
                        particles[:, joint_idx : joint_idx + 1],
                        joint_weights[:, joint_idx],
                    )[0]
                )
            estimate_array = np.asarray(estimate)
            estimates.append(estimate_array)
            ess_per_joint = 1.0 / np.sum(joint_weights * joint_weights, axis=0)
            ess = float(np.mean(ess_per_joint))
            weights = np.mean(joint_weights, axis=1)
            weights = weights / np.sum(weights)
            log_weights = np.log(weights + 1e-300)
        else:
            ll = np.sum(joint_ll, axis=-1)
            weights, log_weights = _normalize_log_weights(log_weights + ll)
            ess = float(1.0 / np.sum(weights * weights))
            estimate_array = chordal_mean(particles, weights)
            estimates.append(estimate_array)
        spread_values.append(_particle_spread_deg(particles, estimate_array))
        ess_values.append(ess)

        should_resample = ess < resample_threshold * num_particles
        resampled_flags.append(should_resample)
        if should_resample and t < t_steps - 1:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            particle_history = [entry[idx] for entry in particle_history]
            log_weights = np.full(
                num_particles, -np.log(num_particles), dtype=np.float64
            )
            log_joint_weights = np.full(
                (num_particles, observations.shape[1]),
                -np.log(num_particles),
                dtype=np.float64,
            )

        if t < t_steps - 1:
            particle_history.append(particles.copy())
            particle_history = particle_history[-history_keep:]

    return ParticleFilterResult(
        estimates=np.asarray(estimates),
        effective_sample_size=np.asarray(ess_values),
        resampled=np.asarray(resampled_flags, dtype=bool),
        particle_spread_deg=np.asarray(spread_values, dtype=np.float64),
    )


def run_filter(
    observations: np.ndarray,
    mask: np.ndarray,
    transition_model: TransitionModel,
    noise_sigma_rad: float,
    num_particles: int,
    rng: np.random.Generator,
    resample_threshold: float = 0.5,
    factorized_update: bool = True,
    proposal_gain: float = 0.2,
    backend: str = "numpy",
    confidence: np.ndarray | None = None,
    joint_noise_sigma_rad: np.ndarray | None = None,
) -> ParticleFilterResult:
    """Run a configured particle filter backend."""
    if backend == "numpy":
        return run_particle_filter(
            observations,
            mask,
            transition_model,
            noise_sigma_rad,
            num_particles,
            rng,
            resample_threshold=resample_threshold,
            factorized_update=factorized_update,
            proposal_gain=proposal_gain,
            confidence=confidence,
            joint_noise_sigma_rad=joint_noise_sigma_rad,
        )
    if backend == "pyrecest":
        from .pyrecest_filter import run_pyrecest_particle_filter

        return run_pyrecest_particle_filter(
            observations,
            mask,
            transition_model,
            noise_sigma_rad,
            num_particles,
            rng,
            resample_threshold=resample_threshold,
            factorized_update=factorized_update,
            proposal_gain=proposal_gain,
            confidence=confidence,
            joint_noise_sigma_rad=joint_noise_sigma_rad,
        )
    raise ValueError(f"unknown filter_backend: {backend}")
