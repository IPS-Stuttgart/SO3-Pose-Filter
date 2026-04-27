"""Cheap SO(3)^K smoothing baselines for noisy pose observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .so3 import chordal_mean, left_apply_delta, left_delta


@dataclass(frozen=True)
class SmootherConfig:
    """Configuration for deterministic smoothing baselines."""

    ema_alpha: float = 0.35
    chordal_window: int = 5


def _validate_inputs(observations: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if observations.ndim != 4 or observations.shape[-2:] != (3, 3):
        raise ValueError(
            f"expected observations shaped (T, J, 3, 3), got {observations.shape}"
        )
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
    """Causal per-joint EMA in the tangent space of the previous estimate.

    Observed joints move by `alpha * log_Rprev(Robs)`. Hidden joints keep the
    previous estimate.
    """

    observations, mask = _validate_inputs(observations, mask)
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")

    t_steps, num_joints = observations.shape[:2]
    estimates = np.empty_like(observations)
    current = _identity_pose(num_joints)

    for t in range(t_steps):
        if t == 0:
            current = np.where(mask[t, :, None, None], observations[t], current)
        else:
            delta = left_delta(current, observations[t])
            updated = left_apply_delta(alpha * delta, current)
            current = np.where(mask[t, :, None, None], updated, current)
        estimates[t] = current

    return estimates


def sliding_chordal_mean_smoother(
    observations: np.ndarray,
    mask: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Offline centered-window chordal mean for each joint.

    Only visible observations inside the local window contribute. If no
    observation for a joint is available in a window, the previous smoothed pose
    is carried forward, with identity as the initial fallback.
    """

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
                current[joint_idx] = chordal_mean(local[:, None, :, :])[0]
            else:
                current[joint_idx] = previous[joint_idx]
        estimates[t] = current
        previous = current

    return estimates


def run_baseline_smoothers(
    observations: np.ndarray,
    mask: np.ndarray,
    config: SmootherConfig | None = None,
) -> dict[str, np.ndarray]:
    """Run all deterministic smoothing baselines used in experiment reports."""

    config = config or SmootherConfig()
    return {
        "smoother_ema": tangent_exponential_smoother(
            observations, mask, alpha=config.ema_alpha
        ),
        "smoother_chordal": sliding_chordal_mean_smoother(
            observations, mask, window=config.chordal_window
        ),
    }
