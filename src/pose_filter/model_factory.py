"""Transition-model factory extensions for experiment runners."""

from __future__ import annotations

from typing import Any

import numpy as np

from .constant_velocity import ConstantVelocityTransition
from .data import PoseSequence
from .transitions import build_transition_model as _build_transition_model
from .transitions import TransitionModel


def _process_noise_cap(process_noise_deg: Any) -> float | None:
    if process_noise_deg is None:
        return None
    return np.radians(float(process_noise_deg))


def build_transition_model(
    name: str,
    train_sequences: list[PoseSequence],
    *,
    process_noise_deg: Any = None,
    config: dict[str, Any] | None = None,
) -> TransitionModel:
    """Build a transition model, including baselines outside ``transitions.py``."""

    if name == "constant_velocity":
        return ConstantVelocityTransition.fit(
            train_sequences,
            max_std_rad=_process_noise_cap(process_noise_deg),
        )
    return _build_transition_model(
        name,
        train_sequences,
        process_noise_deg=process_noise_deg,
        config=config,
    )
