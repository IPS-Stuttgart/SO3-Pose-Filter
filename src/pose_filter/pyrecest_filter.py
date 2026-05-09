"""PyRecEst-backed particle filter adapter for SO(3)^K pose states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .particle_filter import (
    ParticleFilterResult,
    _particle_spread_deg,
    _prepare_confidence,
    _prepare_joint_noise,
    initialize_particles,
    systematic_resample,
)
from .quaternion import quaternions_to_rotations, rotations_to_quaternions
from .so3 import left_apply_delta, left_delta
from .transitions import TransitionModel


def _import_pyrecest_filters():
    try:
        from pyrecest.filters import (
            PartitionedSO3ProductParticleFilter,
            SO3ProductParticleFilter,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional upstream install
        raise ImportError(
            "filter_backend='pyrecest' requires PyRecEst main, or a release with "
            "the public SO3ProductParticleFilter and "
            "PartitionedSO3ProductParticleFilter exports."
        ) from exc
    return SO3ProductParticleFilter, PartitionedSO3ProductParticleFilter


def _import_pyrecest_filter():
    """Return the PyRecEst product filter class for backward-compatible probes."""
    return _import_pyrecest_filters()[0]


def _import_pyrecest_partitioned_filter():
    """Return the PyRecEst partitioned product filter class."""
    return _import_pyrecest_filters()[1]


def is_pyrecest_filter_available() -> bool:
    """Return whether the PyRecEst SO(3)^K particle filter backend is importable."""
    try:
        filter_cls = _import_pyrecest_filter()
    except ImportError:
        return False
    return hasattr(filter_cls, "update_with_geodesic_log_likelihood")


def is_pyrecest_partitioned_filter_available() -> bool:
    """Return whether the PyRecEst partitioned SO(3)^K particle filter is importable."""
    try:
        filter_cls = _import_pyrecest_partitioned_filter()
    except ImportError:
        return False
    return hasattr(filter_cls, "update_with_geodesic_log_likelihood")


def _as_numpy(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _geodesic_update_kwargs(
    observation: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    noise_sigma_rad: float,
    joint_noise_sigma_rad: np.ndarray,
) -> dict[str, object]:
    """Build keyword arguments for PyRecEst's SO(3)^K geodesic update API."""
    return {
        "measurement": rotations_to_quaternions(observation),
        "noise_std": noise_sigma_rad,
        "component_noise_std": joint_noise_sigma_rad,
        "mask": mask.astype(np.float64),
        "confidence": confidence,
        "resample": False,
    }


def _resolve_pyrecest_partition(
    num_joints: int,
    factorized_update: bool,
    particle_blocks,
) -> tuple[tuple[int, ...], ...] | None:
    """Return a PyRecEst partition or ``None`` for a global product filter."""
    if particle_blocks is not None:
        from .block_particle_filter import resolve_particle_blocks

        return resolve_particle_blocks(num_joints, particle_blocks)
    if factorized_update:
        return tuple((joint_idx,) for joint_idx in range(num_joints))
    return None


def _resample_partitioned_filter(
    filter_state,
    particle_history: list[np.ndarray],
    partition: Sequence[Sequence[int]],
    resample_blocks: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Resample selected PyRecEst partition blocks and mirror history arrays."""
    quaternion_particles = _as_numpy(filter_state.particles).copy()
    block_weights = _as_numpy(filter_state.block_weights).copy()
    for block_idx, should_resample in enumerate(resample_blocks):
        if not bool(should_resample):
            continue
        indices = systematic_resample(block_weights[block_idx], rng)
        block = np.asarray(partition[block_idx], dtype=np.int64)
        quaternion_particles[:, block] = quaternion_particles[indices][:, block]
        for history_entry in particle_history:
            history_entry[:, block] = history_entry[indices][:, block]
        block_weights[block_idx] = 1.0 / quaternion_particles.shape[0]
    filter_state.set_particles(quaternion_particles, block_weights=block_weights)


def _resample_product_filter(
    filter_state,
    particle_history: list[np.ndarray],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Resample a global PyRecEst product filter and mirror history arrays."""
    indices = systematic_resample(weights, rng)
    particles = _as_numpy(filter_state.particles)[indices]
    for history_idx, history_entry in enumerate(particle_history):
        particle_history[history_idx] = history_entry[indices]
    filter_state.set_particles(
        particles,
        weights=np.full(
            particles.shape[0],
            1.0 / particles.shape[0],
            dtype=np.float64,
        ),
    )


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
    particle_blocks=None,
) -> ParticleFilterResult:
    """Run the SO(3)^K particle filter using PyRecEst quaternion product particles.

    The surrounding prototype stores rotations as matrices. This adapter keeps
    transition models and metrics in that representation while using PyRecEst's
    ``SO3ProductParticleFilter`` or ``PartitionedSO3ProductParticleFilter`` as
    the particle-state backend. Measurement scoring is delegated to PyRecEst's
    SO(3)^K geodesic log-likelihood helpers, including masks, confidence values,
    heteroskedastic per-joint noise scales, and log-domain weight updates.
    """
    SO3ProductParticleFilter, PartitionedSO3ProductParticleFilter = _import_pyrecest_filters()

    observations = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    confidence = _prepare_confidence(mask, confidence)
    joint_noise_sigma_rad = _prepare_joint_noise(
        noise_sigma_rad,
        mask,
        joint_noise_sigma_rad,
    )
    t_steps, num_joints = observations.shape[:2]
    partition = _resolve_pyrecest_partition(
        num_joints,
        factorized_update,
        particle_blocks,
    )
    is_partitioned = partition is not None

    initial_rotations = initialize_particles(
        observations[0],
        confidence[0] > 0.0,
        num_particles,
        noise_sigma_rad,
        rng,
    )
    initial_quaternions = rotations_to_quaternions(initial_rotations)
    if is_partitioned:
        filter_state = PartitionedSO3ProductParticleFilter(
            int(num_particles),
            int(num_joints),
            partition=partition,
            initial_particles=initial_quaternions,
        )
    else:
        filter_state = SO3ProductParticleFilter(
            int(num_particles),
            int(num_joints),
            initial_particles=initial_quaternions,
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
                particle_history or [particles],
                rng,
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
        if is_partitioned:
            ess_per_block = _as_numpy(
                filter_state.update_with_geodesic_log_likelihood(
                    **_geodesic_update_kwargs(
                        observations[t],
                        mask[t],
                        confidence[t],
                        noise_sigma_rad,
                        joint_noise_sigma_rad[t],
                    )
                )
            ).reshape(-1)
            estimate_array = quaternions_to_rotations(_as_numpy(filter_state.mean()))
            ess = float(np.mean(ess_per_block))
        else:
            ess = filter_state.update_with_geodesic_log_likelihood(
                **_geodesic_update_kwargs(
                    observations[t],
                    mask[t],
                    confidence[t],
                    noise_sigma_rad,
                    joint_noise_sigma_rad[t],
                )
            )
            weights = _as_numpy(filter_state.weights)
            ess = float(np.asarray(ess))
            estimate_array = quaternions_to_rotations(_as_numpy(filter_state.mean()))

        estimates.append(estimate_array)
        spread_values.append(_particle_spread_deg(particles, estimate_array))
        ess_values.append(ess)

        if is_partitioned:
            resample_blocks = ess_per_block < resample_threshold * num_particles
            resampled_flags.append(bool(np.any(resample_blocks)))
            if t < t_steps - 1 and bool(np.any(resample_blocks)):
                if partition is None:
                    raise RuntimeError(
                        "partitioned filters require a joint partition for resampling"
                    )
                _resample_partitioned_filter(
                    filter_state,
                    particle_history,
                    partition,
                    resample_blocks,
                    rng,
                )
        else:
            should_resample = ess < resample_threshold * num_particles
            resampled_flags.append(should_resample)
            if should_resample and t < t_steps - 1:
                _resample_product_filter(filter_state, particle_history, weights, rng)

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
