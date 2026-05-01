"""Constant-velocity transition baseline for SO(3)^K states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import PoseSequence, sequence_pairs
from .so3 import left_apply_delta, left_delta
from .transitions import TransitionModel


@dataclass
class ConstantVelocityTransition(TransitionModel):
    """Per-joint SO(3) constant-velocity transition with residual noise.

    The model uses the previous tangent-space delta as the deterministic velocity.
    If no previous state is available, it falls back to zero velocity, so it can be
    used through the same public transition interface as first-order baselines.
    """

    residual_std: np.ndarray

    name = "constant_velocity"

    @property
    def history_length(self) -> int:
        """Number of previous transition deltas required by the model."""

        return 1

    @classmethod
    def fit(
        cls,
        sequences: list[PoseSequence],
        min_std_rad: float = np.radians(0.25),
        max_std_rad: float | None = None,
    ) -> "ConstantVelocityTransition":
        """Fit residual acceleration noise from consecutive SO(3)^K deltas."""

        residual_chunks = []
        delta_chunks = []
        for seq in sequences:
            rotations = np.asarray(seq.rotations, dtype=np.float64)
            if rotations.shape[0] < 2:
                continue
            deltas = left_delta(rotations[:-1], rotations[1:])
            delta_chunks.append(deltas)
            if deltas.shape[0] >= 2:
                residual_chunks.append(deltas[1:] - deltas[:-1])

        if residual_chunks:
            residuals = np.concatenate(residual_chunks, axis=0)
        elif delta_chunks:
            residuals = np.concatenate(delta_chunks, axis=0)
        else:
            x, y = sequence_pairs(sequences)
            residuals = left_delta(x, y)

        residual_std = np.maximum(np.std(residuals, axis=0), float(min_std_rad))
        if max_std_rad is not None:
            residual_std = np.minimum(residual_std, float(max_std_rad))
        return cls(residual_std=residual_std)

    def _velocity_from_history(self, history: list[np.ndarray]) -> np.ndarray:
        current = np.asarray(history[-1], dtype=np.float64)
        if len(history) < 2:
            return np.zeros(current.shape[:-2] + (3,), dtype=np.float64)
        previous = np.asarray(history[-2], dtype=np.float64)
        return left_delta(previous, current)

    def sample_next(
        self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None
    ) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=np.float64)
        if n_samples is not None:
            base = np.repeat(x_k[None, ...], int(n_samples), axis=0)
            noise = rng.normal(0.0, self.residual_std, size=base.shape[:-2] + (3,))
            return left_apply_delta(noise, base)
        noise = rng.normal(0.0, self.residual_std, size=x_k.shape[:-2] + (3,))
        return left_apply_delta(noise, x_k)

    def sample_next_from_history(
        self, history: list[np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
        velocity = self._velocity_from_history(history)
        noise = rng.normal(0.0, self.residual_std, size=velocity.shape)
        return left_apply_delta(velocity + noise, history[-1])

    def deterministic_next(self, x_k: np.ndarray) -> np.ndarray:
        return np.asarray(x_k, dtype=np.float64).copy()

    def deterministic_next_from_history(self, history: list[np.ndarray]) -> np.ndarray:
        return left_apply_delta(self._velocity_from_history(history), history[-1])

    def log_prob_next(self, x_next: np.ndarray, x_k: np.ndarray) -> np.ndarray:
        delta = left_delta(x_k, x_next)
        z = delta / self.residual_std
        return -0.5 * np.sum(
            z * z + np.log(2.0 * np.pi * self.residual_std * self.residual_std),
            axis=(-1, -2),
        )

    def log_prob_next_from_history(
        self, x_next: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        delta = left_delta(history[-1], x_next)
        velocity = self._velocity_from_history(history)
        z = (delta - velocity) / self.residual_std
        return -0.5 * np.sum(
            z * z + np.log(2.0 * np.pi * self.residual_std * self.residual_std),
            axis=(-1, -2),
        )
