"""Block particle filter approximation for product rotation states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .measurements import joint_log_likelihood
from .particle_filter import (
    ParticleFilterResult,
    _normalize_log_weights,
    _particle_spread_deg,
    _prepare_confidence,
    _prepare_joint_noise,
    initialize_particles,
    systematic_resample,
)
from .so3 import chordal_mean, left_apply_delta, left_delta
from .transitions import TransitionModel


ParticleBlocks = str | Sequence[Sequence[int]]

# The 23 local SMPL body joints used by this project are pose parameters 1..23
# after removing the global root orientation. Zero-based indices here follow
# the resulting SO(3)^23 state vector:
#   0 L_hip, 1 R_hip, 2 spine1, 3 L_knee, 4 R_knee, 5 spine2,
#   6 L_ankle, 7 R_ankle, 8 spine3, 9 L_foot, 10 R_foot,
#   11 neck, 12 L_collar, 13 R_collar, 14 head,
#   15 L_shoulder, 16 R_shoulder, 17 L_elbow, 18 R_elbow,
#   19 L_wrist, 20 R_wrist, 21 L_hand, 22 R_hand.
DEFAULT_SMPL_BODY_BLOCKS: tuple[tuple[int, ...], ...] = (
    (2, 5, 8, 11, 14),  # torso, neck, head
    (0, 3, 6, 9),  # left leg
    (1, 4, 7, 10),  # right leg
    (12, 15, 17, 19, 21),  # left shoulder chain / arm
    (13, 16, 18, 20, 22),  # right shoulder chain / arm
)


def _validate_blocks(
    blocks: Sequence[Sequence[int]], num_joints: int
) -> tuple[tuple[int, ...], ...]:
    if num_joints <= 0:
        raise ValueError("num_joints must be positive")
    normalized: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for block_idx, raw_block in enumerate(blocks):
        block = tuple(int(joint_idx) for joint_idx in raw_block)
        if not block:
            raise ValueError(f"particle block {block_idx} is empty")
        for joint_idx in block:
            if joint_idx < 0 or joint_idx >= num_joints:
                raise ValueError(
                    f"particle block {block_idx} contains joint {joint_idx}, "
                    f"but valid joints are 0..{num_joints - 1}"
                )
            if joint_idx in seen:
                raise ValueError(
                    f"joint {joint_idx} appears in more than one particle block"
                )
            seen.add(joint_idx)
        normalized.append(block)

    missing = sorted(set(range(num_joints)) - seen)
    if missing:
        raise ValueError(
            "particle blocks must cover every joint exactly once; "
            f"missing {missing}"
        )
    return tuple(normalized)


def _contiguous_blocks(
    num_joints: int, block_size: int = 4
) -> tuple[tuple[int, ...], ...]:
    block_size = max(1, int(block_size))
    return tuple(
        tuple(range(start, min(start + block_size, num_joints)))
        for start in range(0, num_joints, block_size)
    )


def resolve_particle_blocks(
    num_joints: int,
    particle_blocks: ParticleBlocks | None,
) -> tuple[tuple[int, ...], ...]:
    """Resolve a named or explicit block partition.

    The returned blocks form a strict partition of the joint axis. This is
    important because block-wise resampling creates a product approximation over
    blocks; overlapping blocks would double-count measurements, and missing
    joints would have undefined weights.
    """
    if particle_blocks is None:
        return (tuple(range(num_joints)),)

    if isinstance(particle_blocks, str):
        name = particle_blocks.strip().lower().replace("-", "_")
        if name in {"", "none", "global", "full"}:
            return (tuple(range(num_joints)),)
        if name in {"joint", "joints", "singleton", "singletons", "factorized"}:
            return tuple((joint_idx,) for joint_idx in range(num_joints))
        if name in {"smpl", "smpl_body", "body"}:
            if num_joints != 23:
                raise ValueError(
                    "particle_blocks='smpl_body' is defined for the 23 local SMPL "
                    f"body joints, got num_joints={num_joints}"
                )
            return _validate_blocks(DEFAULT_SMPL_BODY_BLOCKS, num_joints)
        if name in {"auto", "contiguous"}:
            if num_joints == 23:
                return _validate_blocks(DEFAULT_SMPL_BODY_BLOCKS, num_joints)
            return _validate_blocks(_contiguous_blocks(num_joints), num_joints)
        raise ValueError(
            "particle_blocks must be one of 'smpl_body', 'auto', 'contiguous', "
            "'joint', 'global', or an explicit list of joint-index lists"
        )

    return _validate_blocks(particle_blocks, num_joints)


def format_particle_blocks(particle_blocks: ParticleBlocks | None) -> str:
    """Return a compact CSV/JSON-friendly representation of block settings."""
    if particle_blocks is None:
        return ""
    if isinstance(particle_blocks, str):
        return particle_blocks
    return ";".join(
        ",".join(str(int(joint_idx)) for joint_idx in block)
        for block in particle_blocks
    )


def _resample_block_in_place(
    particles: np.ndarray,
    particle_history: list[np.ndarray],
    block: Sequence[int],
    indices: np.ndarray,
) -> None:
    block_indices = np.asarray(block, dtype=np.int64)
    particles[:, block_indices] = particles[indices][:, block_indices]
    for history_entry in particle_history:
        history_entry[:, block_indices] = history_entry[indices][:, block_indices]


def run_block_particle_filter(
    observations: np.ndarray,
    mask: np.ndarray,
    transition_model: TransitionModel,
    noise_sigma_rad: float,
    num_particles: int,
    rng: np.random.Generator,
    *,
    particle_blocks: ParticleBlocks = "smpl_body",
    resample_threshold: float = 0.5,
    proposal_gain: float = 0.2,
    confidence: np.ndarray | None = None,
    joint_noise_sigma_rad: np.ndarray | None = None,
    outlier_prob: float = 0.0,
) -> ParticleFilterResult:
    """Run a block particle filter on one SO(3)^K sequence.

    This is a middle ground between the existing global particle filter and the
    per-joint factorized update. Each block keeps one weight vector and is
    resampled independently, preserving dependencies inside a body block while
    avoiding the severe weight collapse of one global SO(3)^23 weight vector.

    Resampling a block independently creates hybrid full-body particles assembled
    from different source particles. That is the intended product-of-blocks
    approximation; it is strongest when transitions and measurement errors are
    mostly local to the configured blocks.
    """
    observations = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if observations.ndim != 4 or observations.shape[-2:] != (3, 3):
        raise ValueError("observations must have shape [T, J, 3, 3]")
    if mask.shape != observations.shape[:-2]:
        raise ValueError(
            f"expected mask shaped {observations.shape[:-2]}, got {mask.shape}"
        )

    confidence = _prepare_confidence(mask, confidence)
    joint_noise_sigma_rad = _prepare_joint_noise(
        noise_sigma_rad, mask, joint_noise_sigma_rad
    )
    t_steps, num_joints = observations.shape[:2]
    blocks = resolve_particle_blocks(num_joints, particle_blocks)
    block_count = len(blocks)

    particles = initialize_particles(
        observations[0], confidence[0] > 0.0, num_particles, noise_sigma_rad, rng
    )
    log_block_weights = np.full(
        (block_count, int(num_particles)), -np.log(num_particles), dtype=np.float64
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
            particles = left_apply_delta(
                correction_weight * delta_to_observation, particles
            )

        joint_ll = joint_log_likelihood(
            observations[t],
            particles,
            mask[t],
            noise_sigma_rad,
            confidence=confidence[t],
            joint_noise_sigma_rad=joint_noise_sigma_rad[t],
            outlier_prob=outlier_prob,
        )

        estimate_array = np.empty_like(observations[t])
        block_weights: list[np.ndarray] = []
        ess_per_block = np.empty(block_count, dtype=np.float64)
        for block_idx, block in enumerate(blocks):
            block_ll = np.sum(joint_ll[:, block], axis=-1)
            weights, log_block_weights[block_idx] = _normalize_log_weights(
                log_block_weights[block_idx] + block_ll
            )
            block_weights.append(weights)
            ess_per_block[block_idx] = 1.0 / np.sum(weights * weights)
            for joint_idx in block:
                estimate_array[joint_idx] = chordal_mean(
                    particles[:, joint_idx : joint_idx + 1], weights
                )[0]

        estimates.append(estimate_array)
        ess = float(np.mean(ess_per_block))
        ess_values.append(ess)
        spread_values.append(_particle_spread_deg(particles, estimate_array))

        resample_blocks = ess_per_block < resample_threshold * num_particles
        resampled_flags.append(bool(np.any(resample_blocks)))
        if t < t_steps - 1:
            for block_idx, should_resample in enumerate(resample_blocks):
                if not bool(should_resample):
                    continue
                indices = systematic_resample(block_weights[block_idx], rng)
                _resample_block_in_place(
                    particles, particle_history, blocks[block_idx], indices
                )
                log_block_weights[block_idx] = -np.log(num_particles)

        if t < t_steps - 1:
            particle_history.append(particles.copy())
            particle_history = particle_history[-history_keep:]

    return ParticleFilterResult(
        estimates=np.asarray(estimates),
        effective_sample_size=np.asarray(ess_values),
        resampled=np.asarray(resampled_flags, dtype=bool),
        particle_spread_deg=np.asarray(spread_values, dtype=np.float64),
    )
