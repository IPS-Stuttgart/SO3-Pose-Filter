from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyrecest.filters import ManifoldExponentialMovingAverage  # type: ignore[import-untyped]
from pyrecest.smoothers import SO3ChordalMeanSmoother  # type: ignore[import-untyped]

from .so3 import left_apply_delta, left_delta


@dataclass(frozen=True)
class SmootherConfig:
    ema_alpha: float = 0.35
    chordal_window: int = 5


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


def run_baseline_smoothers(
    observations: np.ndarray,
    mask: np.ndarray,
    config: SmootherConfig | None = None,
) -> dict[str, np.ndarray]:
    config = config or SmootherConfig()
    return {
        "smoother_ema": tangent_exponential_smoother(observations, mask, alpha=config.ema_alpha),
        "smoother_chordal": sliding_chordal_mean_smoother(observations, mask, window=config.chordal_window),
    }
