"""Particle-filter runner with an optional measurement-conditioned proposal."""

from __future__ import annotations

import numpy as np

from .particle_filter import (
    ParticleFilterResult,
    _normalize_log_weights,
    _normalize_log_weights_axis0,
    _particle_spread_deg,
    _prepare_confidence,
    _prepare_joint_noise,
    initialize_particles,
    systematic_resample,
)
from .proposals import ProposalModel
from .so3 import chordal_mean, geodesic_distance, left_apply_delta, left_delta
from .transitions import TransitionModel


def run_particle_filter_with_proposal(
    observations: np.ndarray,
    mask: np.ndarray,
    transition_model: TransitionModel,
    noise_sigma_rad: float,
    num_particles: int,
    rng: np.random.Generator,
    *,
    proposal_model: ProposalModel | None = None,
    resample_threshold: float = 0.5,
    factorized_update: bool = True,
    proposal_gain: float = 0.2,
    confidence: np.ndarray | None = None,
    joint_noise_sigma_rad: np.ndarray | None = None,
) -> ParticleFilterResult:
    """Run an SO(3)^K particle filter with a learned proposal correction.

    The proposal is applied after transition prediction and before the standard
    guided correction / measurement-likelihood update. With ``proposal_model=None``
    this is equivalent to the existing NumPy particle-filter path.
    """
    observations = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    confidence = _prepare_confidence(mask, confidence)
    joint_noise_sigma_rad = _prepare_joint_noise(
        noise_sigma_rad,
        mask,
        joint_noise_sigma_rad,
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
    transition_history_keep = int(getattr(transition_model, "history_length", 0)) + 1
    proposal_history_keep = int(getattr(proposal_model, "history_length", 0)) + 1
    history_keep = max(transition_history_keep, proposal_history_keep)

    for t in range(t_steps):
        if t > 0:
            particles = transition_model.sample_next_from_history(
                particle_history or [particles], rng
            )

        if proposal_model is not None:
            particles = proposal_model.propose_next(
                particles,
                particle_history or [particles],
                observations[t],
                mask[t],
                confidence[t],
                joint_noise_sigma_rad[t],
                rng,
            )

        if proposal_gain > 0.0:
            delta_to_observation = left_delta(particles, observations[t])
            correction_weight = float(proposal_gain) * confidence[t][None, :, None]
            particles = left_apply_delta(correction_weight * delta_to_observation, particles)

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
            log_weights = np.full(num_particles, -np.log(num_particles), dtype=np.float64)
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
