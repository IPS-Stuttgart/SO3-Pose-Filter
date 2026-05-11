from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyrecest.filters import (
    ManifoldExponentialMovingAverage,  # type: ignore[import-untyped]
)
from pyrecest.smoothers import SO3ChordalMeanSmoother  # type: ignore[import-untyped]

from .so3 import left_apply_delta, left_delta


@dataclass(frozen=True)
class SmootherConfig:
    ema_alpha: float = 0.35
    chordal_window: int = 5
    tangent_savgol_window: int = 7
    tangent_savgol_degree: int = 2


def _validate_inputs(observations: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if observations.ndim != 4 or observations.shape[-2:] != (3, 3):
        raise ValueError(f"expected observations shaped (T, J, 3, 3), got {observations.shape}")
    if mask.shape != observations.shape[:2]:
        raise ValueError(f"expected mask shaped {observations.shape[:2]}, got {mask.shape}")
    return observations, mask


def _identity_pose(num_joints: int) -> np.ndarray:
    return np.broadcast_to(np.eye(3), (num_joints, 3, 3)).copy()


def tangent_exponential_smoother(
    observations: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    """Causal per-joint PyRecEst manifold EMA over visible observations."""

    observations, mask = _validate_inputs(observations, mask)
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")

    t_steps, num_joints = observations.shape[:2]
    estimates = np.empty_like(observations)
    previous = _identity_pose(num_joints)
    ema = ManifoldExponentialMovingAverage(
        initial_state=None,
        alpha=alpha,
        phi=lambda rotations, delta: left_apply_delta(delta, rotations),
        phi_inv=left_delta,
    )

    for t in range(t_steps):
        sample = np.where(mask[t, :, None, None], observations[t], previous)
        ema.update(sample)
        previous = np.asarray(ema.get_point_estimate(), dtype=np.float64)
        estimates[t] = previous

    return estimates


def sliding_chordal_mean_smoother(
    observations: np.ndarray,
    mask: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Offline per-joint PyRecEst chordal mean over visible local windows."""

    observations, mask = _validate_inputs(observations, mask)
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    t_steps, num_joints = observations.shape[:2]
    half = window // 2
    estimates = np.empty_like(observations)
    previous = _identity_pose(num_joints)

    for t in range(t_steps):
        start = max(0, t - half)
        stop = min(t_steps, t + half + 1)
        current = np.empty((num_joints, 3, 3), dtype=np.float64)
        for joint_idx in range(num_joints):
            visible = mask[start:stop, joint_idx]
            if np.any(visible):
                local = observations[start:stop, joint_idx][visible]
                current[joint_idx] = SO3ChordalMeanSmoother.chordal_mean(local)
            else:
                current[joint_idx] = previous[joint_idx]
        estimates[t] = current
        previous = current

    return estimates


def _nearest_visible_index(mask: np.ndarray, target: int) -> int | None:
    visible = np.flatnonzero(mask)
    if visible.size == 0:
        return None
    return int(visible[np.argmin(np.abs(visible - target))])


def _local_polynomial_at_zero(offsets: np.ndarray, values: np.ndarray, degree: int) -> np.ndarray:
    degree = min(int(degree), max(0, offsets.size - 1))
    if degree == 0:
        return values[np.argmin(np.abs(offsets))]

    scale = max(float(np.max(np.abs(offsets))), 1.0)
    x = offsets.astype(np.float64) / scale
    vandermonde = np.vander(x, N=degree + 1, increasing=True)
    weights = (1.0 - np.clip(np.abs(x), 0.0, 1.0) ** 3) ** 3
    weights = np.maximum(weights, 1e-6)
    weighted_design = vandermonde * np.sqrt(weights)[:, None]
    weighted_values = values * np.sqrt(weights)[:, None]
    coeffs, *_ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
    return coeffs[0]


def tangent_savgol_smoother(
    observations: np.ndarray,
    mask: np.ndarray,
    window: int = 7,
    degree: int = 2,
) -> np.ndarray:
    """Offline per-joint local polynomial smoother in SO(3) tangent spaces.

    This is a Savitzky-Golay-style temporal baseline: for each frame and joint,
    visible observations in a centered local window are mapped to the tangent
    space of the nearest visible rotation, a weighted polynomial is fitted over
    time offsets, and the intercept is mapped back to SO(3). Missing windows
    fall back to the previous estimate for that joint.
    """

    observations, mask = _validate_inputs(observations, mask)
    window = int(window)
    degree = int(degree)
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if degree < 0:
        raise ValueError("degree must be non-negative")

    t_steps, num_joints = observations.shape[:2]
    half = window // 2
    estimates = np.empty_like(observations)
    previous = _identity_pose(num_joints)

    for t in range(t_steps):
        start = max(0, t - half)
        stop = min(t_steps, t + half + 1)
        offsets_full = np.arange(start, stop, dtype=np.float64) - float(t)
        current = np.empty((num_joints, 3, 3), dtype=np.float64)
        for joint_idx in range(num_joints):
            visible = mask[start:stop, joint_idx]
            if not np.any(visible):
                current[joint_idx] = previous[joint_idx]
                continue
            local_indices = np.arange(start, stop)[visible]
            anchor_idx = _nearest_visible_index(mask[:, joint_idx], t)
            if anchor_idx is None:
                current[joint_idx] = previous[joint_idx]
                continue
            anchor = observations[anchor_idx, joint_idx]
            deltas = left_delta(anchor, observations[local_indices, joint_idx])
            delta_at_t = _local_polynomial_at_zero(offsets_full[visible], deltas, degree)
            current[joint_idx] = left_apply_delta(delta_at_t, anchor)
        estimates[t] = current
        previous = current

    return estimates


def run_baseline_smoothers(
    observations: np.ndarray,
    mask: np.ndarray,
    config: SmootherConfig | None = None,
) -> dict[str, np.ndarray]:
    config = config or SmootherConfig()
    return {
        "smoother_ema": tangent_exponential_smoother(observations, mask, alpha=config.ema_alpha),
        "smoother_chordal": sliding_chordal_mean_smoother(observations, mask, window=config.chordal_window),
        "savgol_tangent": tangent_savgol_smoother(
            observations,
            mask,
            window=config.tangent_savgol_window,
            degree=config.tangent_savgol_degree,
        ),
    }
