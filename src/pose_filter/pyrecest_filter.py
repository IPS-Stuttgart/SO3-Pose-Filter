"""PyRecEst-backed particle filter adapter for SO(3)^K pose states."""

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
from .quaternion import quaternions_to_rotations, rotations_to_quaternions
from .so3 import chordal_mean, geodesic_distance, left_apply_delta, left_delta
from .transitions import TransitionModel


def _import_pyrecest_filter():
    try:
        from pyrecest.filters import SO3ProductParticleFilter
    except ImportError as exc:
        try:
            from pyrecest.filters.so3_product_particle_filter import (
                SO3ProductParticleFilter,
            )
        except ImportError:
            raise ImportError(
                "filter_backend='pyrecest' requires PyRecEst with "
                "pyrecest.filters.SO3ProductParticleFilter available."
            ) from exc
    return SO3ProductParticleFilter


def is_pyrecest_filter_available() -> bool:
    """Return whether the PyRecEst SO(3)^K particle filter backend is importable."""
    try:
        _import_pyrecest_filter()
    except ImportError:
        return False
    return True


def _as_numpy(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _component_geodesic_log_likelihood(
    filter_state,
    observation: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    noise_sigma_rad: float,
    joint_noise_sigma_rad: np.ndarray,
) -> np.ndarray:
    """Evaluate PyRecEst's component SO(3)^K geodesic log likelihood.

    The fallback keeps the adapter usable with released PyRecEst versions while
    the primary path exercises PyRecEst main's confidence-aware, heteroskedastic
    component likelihood API.
    """
    if hasattr(filter_state, "component_geodesic_log_likelihood"):
        return _as_numpy(
            filter_state.component_geodesic_log_likelihood(
                rotations_to_quaternions(observation),
                noise_sigma_rad,
                component_noise_std=joint_noise_sigma_rad,
                mask=mask.astype(np.float64),
                confidence=confidence,
            )
        )

    particles = quaternions_to_rotations(_as_numpy(filter_state.particles))
    dist = geodesic_distance(particles, observation)
    return -0.5 * confidence[None, :] * (dist / joint_noise_sigma_rad[None, :]) ** 2


def _update_with_geodesic_log_likelihood(
    filter_state,
    observation: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    noise_sigma_rad: float,
    joint_noise_sigma_rad: np.ndarray,
    log_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Update PyRecEst weights using its geodesic log-likelihood API if present."""
    if hasattr(filter_state, "update_with_geodesic_log_likelihood"):
        ess = filter_state.update_with_geodesic_log_likelihood(
            rotations_to_quaternions(observation),
            noise_sigma_rad,
            component_noise_std=joint_noise_sigma_rad,
            mask=mask.astype(np.float64),
            confidence=confidence,
            resample=False,
        )
        weights = _as_numpy(filter_state.weights)
        return weights, np.log(weights + 1e-300), float(np.asarray(ess))

    joint_ll = _component_geodesic_log_likelihood(
        filter_state,
        observation,
        mask,
        confidence,
        noise_sigma_rad,
        joint_noise_sigma_rad,
    )
    weights, log_weights = _normalize_log_weights(log_weights + np.sum(joint_ll, axis=-1))
    filter_state.set_particles(_as_numpy(filter_state.particles), weights=weights)
    return weights, log_weights, float(np.asarray(filter_state.effective_sample_size()))


def run_pyrecest_particle_filter(
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
    """Run the SO(3)^K particle filter using PyRecEst quaternion product particles.

    The surrounding prototype stores rotations as matrices. This adapter keeps
    transition models and metrics in that representation while using PyRecEst's
    ``SO3ProductParticleFilter`` as the particle-state backend.  Measurement
    scoring is delegated to PyRecEst's SO(3)^K geodesic log-likelihood helpers
    when available, including masks, confidence values, and heteroskedastic
    per-joint noise scales.
    """
    SO3ProductParticleFilter = _import_pyrecest_filter()

    observations = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    confidence = _prepare_confidence(mask, confidence)
    joint_noise_sigma_rad = _prepare_joint_noise(
        noise_sigma_rad, mask, joint_noise_sigma_rad
    )
    t_steps, num_joints = observations.shape[:2]
    initial_rotations = initialize_particles(
        observations[0], confidence[0] > 0.0, num_particles, noise_sigma_rad, rng
    )
    filter_state = SO3ProductParticleFilter(
        int(num_particles),
        int(num_joints),
        initial_particles=rotations_to_quaternions(initial_rotations),
    )

    log_weights = np.full(num_particles, -np.log(num_particles), dtype=np.float64)
    log_joint_weights = np.full(
        (num_particles, num_joints), -np.log(num_particles), dtype=np.float64
    )
    estimates = []
    ess_values = []
    resampled_flags = []
    spread_values = []
    particle_history: list[np.ndarray] = []
    history_keep = int(getattr(transition_model, "history_length", 0)) + 1

    for t in range(t_steps):
        if t > 0:
            particles = quaternions_to_rotations(_as_numpy(filter_state.particles))
            predicted = transition_model.sample_next_from_history(
                particle_history or [particles], rng
            )
            filter_state.set_particles(rotations_to_quaternions(predicted))

        if proposal_gain > 0.0:
            particles = quaternions_to_rotations(_as_numpy(filter_state.particles))
            delta_to_observation = left_delta(particles, observations[t])
            correction_weight = float(proposal_gain) * confidence[t][None, :, None]
            correction = correction_weight * delta_to_observation
            corrected = left_apply_delta(correction, particles)
            filter_state.set_particles(rotations_to_quaternions(corrected))

        particles = quaternions_to_rotations(_as_numpy(filter_state.particles))
        joint_ll = _component_geodesic_log_likelihood(
            filter_state,
            observations[t],
            mask[t],
            confidence[t],
            noise_sigma_rad,
            joint_noise_sigma_rad[t],
        )

        if factorized_update:
            joint_weights, log_joint_weights = _normalize_log_weights_axis0(
                log_joint_weights + joint_ll
            )
            estimate = []
            for joint_idx in range(num_joints):
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
            filter_state.set_particles(
                _as_numpy(filter_state.particles), weights=weights
            )
        else:
            weights, log_weights, ess = _update_with_geodesic_log_likelihood(
                filter_state,
                observations[t],
                mask[t],
                confidence[t],
                noise_sigma_rad,
                joint_noise_sigma_rad[t],
                log_weights,
            )
            estimate_array = quaternions_to_rotations(_as_numpy(filter_state.mean()))
            estimates.append(estimate_array)

        spread_values.append(_particle_spread_deg(particles, estimate_array))
        ess_values.append(ess)
        should_resample = ess < resample_threshold * num_particles
        resampled_flags.append(should_resample)
        if should_resample and t < t_steps - 1:
            idx = systematic_resample(weights, rng)
            particles = _as_numpy(filter_state.particles)[idx]
            particle_history = [entry[idx] for entry in particle_history]
            filter_state.set_particles(
                particles,
                weights=np.full(num_particles, 1.0 / num_particles, dtype=np.float64),
            )
            log_weights = np.full(
                num_particles, -np.log(num_particles), dtype=np.float64
            )
            log_joint_weights = np.full(
                (num_particles, num_joints), -np.log(num_particles), dtype=np.float64
            )

        if t < t_steps - 1:
            particle_history.append(
                quaternions_to_rotations(_as_numpy(filter_state.particles))
            )
            particle_history = particle_history[-history_keep:]

    return ParticleFilterResult(
        estimates=np.asarray(estimates),
        effective_sample_size=np.asarray(ess_values),
        resampled=np.asarray(resampled_flags, dtype=bool),
        particle_spread_deg=np.asarray(spread_values, dtype=np.float64),
    )
