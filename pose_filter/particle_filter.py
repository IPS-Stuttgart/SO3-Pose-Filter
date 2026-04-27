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


def _normalize_log_weights_axis0(log_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = log_weights - np.max(log_weights, axis=0, keepdims=True)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights, axis=0, keepdims=True)
    return weights, np.log(weights + 1e-300)


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
) -> ParticleFilterResult:
    """Run a guided bootstrap particle filter on one sequence.

    `proposal_gain` applies a small SO(3) correction toward observed joints before
    weighting. This keeps the low-particle prototype useful in the high-dimensional
    product space without changing the measurement likelihood used for scoring.
    """
    observations = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    t_steps = observations.shape[0]
    particles = initialize_particles(
        observations[0], mask[0], num_particles, noise_sigma_rad, rng
    )
    log_weights = np.full(num_particles, -np.log(num_particles), dtype=np.float64)
    log_joint_weights = np.full(
        (num_particles, observations.shape[1]), -np.log(num_particles), dtype=np.float64
    )
    estimates = []
    ess_values = []
    resampled_flags = []

    for t in range(t_steps):
        if t > 0:
            particles = transition_model.sample_next(particles, rng)

        if proposal_gain > 0.0:
            delta_to_observation = left_delta(particles, observations[t])
            correction = np.where(
                mask[t][None, :, None], float(proposal_gain) * delta_to_observation, 0.0
            )
            particles = left_apply_delta(correction, particles)

        dist = geodesic_distance(particles, observations[t])
        joint_ll = -0.5 * (dist / max(noise_sigma_rad, 1e-8)) ** 2
        joint_ll = np.where(mask[t][None, :], joint_ll, 0.0)

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
            estimates.append(np.asarray(estimate))
            ess_per_joint = 1.0 / np.sum(joint_weights * joint_weights, axis=0)
            ess = float(np.mean(ess_per_joint))
            weights = np.mean(joint_weights, axis=1)
            weights = weights / np.sum(weights)
            log_weights = np.log(weights + 1e-300)
        else:
            ll = np.sum(joint_ll, axis=-1)
            weights, log_weights = _normalize_log_weights(log_weights + ll)
            ess = float(1.0 / np.sum(weights * weights))
            estimates.append(chordal_mean(particles, weights))
        ess_values.append(ess)

        should_resample = ess < resample_threshold * num_particles
        resampled_flags.append(should_resample)
        if should_resample and t < t_steps - 1:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            log_weights = np.full(num_particles, -np.log(num_particles), dtype=np.float64)
            log_joint_weights = np.full(
                (num_particles, observations.shape[1]), -np.log(num_particles), dtype=np.float64
            )

    return ParticleFilterResult(
        estimates=np.asarray(estimates),
        effective_sample_size=np.asarray(ess_values),
        resampled=np.asarray(resampled_flags, dtype=bool),
    )
