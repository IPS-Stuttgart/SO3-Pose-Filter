"""Transition baselines for SO(3)^K states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import PoseSequence, sequence_pairs
from .so3 import geodesic_distance, left_apply_delta, left_delta, log_map


class TransitionModel:
    name = "base"

    def sample_next(self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def deterministic_next(self, x_k: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def log_prob_next(self, x_next: np.ndarray, x_k: np.ndarray) -> np.ndarray | None:
        return None


class PersistenceTransition(TransitionModel):
    name = "persistence"

    def sample_next(self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=np.float64)
        if n_samples is None:
            return x_k.copy()
        return np.repeat(x_k[None, ...], n_samples, axis=0)

    def deterministic_next(self, x_k: np.ndarray) -> np.ndarray:
        return np.asarray(x_k, dtype=np.float64).copy()


@dataclass
class GaussianRandomWalkTransition(TransitionModel):
    mean_delta: np.ndarray
    std_delta: np.ndarray

    name = "gaussian_rw"

    @classmethod
    def fit(
        cls,
        sequences: list[PoseSequence],
        min_std_rad: float = np.radians(0.25),
        max_std_rad: float | None = None,
    ) -> "GaussianRandomWalkTransition":
        x, y = sequence_pairs(sequences)
        deltas = left_delta(x, y)
        mean = np.mean(deltas, axis=0)
        std = np.maximum(np.std(deltas, axis=0), min_std_rad)
        if max_std_rad is not None:
            std = np.minimum(std, float(max_std_rad))
        return cls(mean_delta=mean, std_delta=std)

    def _sample_delta(self, base_shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean_delta, self.std_delta, size=base_shape + self.mean_delta.shape)

    def sample_next(self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=np.float64)
        if n_samples is not None:
            base = np.repeat(x_k[None, ...], n_samples, axis=0)
            delta = self._sample_delta((n_samples,), rng)
            return left_apply_delta(delta, base)
        delta = rng.normal(self.mean_delta, self.std_delta, size=x_k.shape[:-2] + (3,))
        return left_apply_delta(delta, x_k)

    def deterministic_next(self, x_k: np.ndarray) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=np.float64)
        delta = np.broadcast_to(self.mean_delta, x_k.shape[:-2] + (3,))
        return left_apply_delta(delta, x_k)

    def log_prob_next(self, x_next: np.ndarray, x_k: np.ndarray) -> np.ndarray:
        delta = left_delta(x_k, x_next)
        z = (delta - self.mean_delta) / self.std_delta
        return -0.5 * np.sum(z * z + np.log(2.0 * np.pi * self.std_delta * self.std_delta), axis=(-1, -2))


@dataclass
class LearnedDeltaTransition(TransitionModel):
    weights: np.ndarray
    residual_std: np.ndarray
    ridge: float = 1e-3

    name = "learned_delta"

    @classmethod
    def fit(
        cls,
        sequences: list[PoseSequence],
        ridge: float = 1e-3,
        min_std_rad: float = np.radians(0.25),
        max_std_rad: float | None = None,
    ) -> "LearnedDeltaTransition":
        x, y = sequence_pairs(sequences)
        x_features = log_map(x).reshape(x.shape[0], -1)
        targets = left_delta(x, y).reshape(x.shape[0], -1)
        design = np.concatenate([x_features, np.ones((x_features.shape[0], 1))], axis=1)
        reg = ridge * np.eye(design.shape[1])
        reg[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + reg, design.T @ targets)
        residual = targets - design @ weights
        residual_std = np.maximum(np.std(residual, axis=0).reshape(x.shape[1], 3), min_std_rad)
        if max_std_rad is not None:
            residual_std = np.minimum(residual_std, float(max_std_rad))
        return cls(weights=weights, residual_std=residual_std, ridge=ridge)

    def _mean_delta(self, x_k: np.ndarray) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=np.float64)
        features = log_map(x_k).reshape(-1, x_k.shape[-3] * 3)
        design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
        pred = design @ self.weights
        return pred.reshape(x_k.shape[:-2] + (3,))

    def sample_next(self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=np.float64)
        if n_samples is not None:
            base = np.repeat(x_k[None, ...], n_samples, axis=0)
            mean = self._mean_delta(base)
            noise = rng.normal(0.0, self.residual_std, size=mean.shape)
            return left_apply_delta(mean + noise, base)
        mean = self._mean_delta(x_k)
        noise = rng.normal(0.0, self.residual_std, size=mean.shape)
        return left_apply_delta(mean + noise, x_k)

    def deterministic_next(self, x_k: np.ndarray) -> np.ndarray:
        return left_apply_delta(self._mean_delta(x_k), x_k)

    def log_prob_next(self, x_next: np.ndarray, x_k: np.ndarray) -> np.ndarray:
        delta = left_delta(x_k, x_next)
        mean = self._mean_delta(x_k)
        z = (delta - mean) / self.residual_std
        return -0.5 * np.sum(z * z + np.log(2.0 * np.pi * self.residual_std * self.residual_std), axis=(-1, -2))


def build_transition_model(
    name: str,
    train_sequences: list[PoseSequence],
    process_noise_deg: float | None = None,
) -> TransitionModel:
    """Fit or construct a transition model by config name."""
    max_std_rad = None if process_noise_deg is None else np.radians(float(process_noise_deg))
    if name == "persistence":
        return PersistenceTransition()
    if name == "gaussian_rw":
        return GaussianRandomWalkTransition.fit(train_sequences, max_std_rad=max_std_rad)
    if name == "learned_delta":
        return LearnedDeltaTransition.fit(train_sequences, max_std_rad=max_std_rad)
    raise ValueError(f"unknown transition_model: {name}")


def one_step_error_deg(model: TransitionModel, sequences: list[PoseSequence]) -> float:
    """Mean one-step deterministic prediction error in degrees."""
    vals = []
    for seq in sequences:
        if seq.rotations.shape[0] < 2:
            continue
        pred = model.deterministic_next(seq.rotations[:-1])
        vals.append(geodesic_distance(pred, seq.rotations[1:]).reshape(-1))
    if not vals:
        return float("nan")
    return float(np.degrees(np.mean(np.concatenate(vals))))


def rollout_error_deg(model: TransitionModel, sequences: list[PoseSequence], horizon: int) -> float:
    """Mean deterministic rollout error in degrees over a fixed horizon."""
    errors = []
    horizon = int(horizon)
    for seq in sequences:
        x = seq.rotations[0]
        max_t = min(horizon, seq.rotations.shape[0] - 1)
        for t in range(1, max_t + 1):
            x = model.deterministic_next(x)
            errors.append(geodesic_distance(x, seq.rotations[t]).reshape(-1))
    if not errors:
        return float("nan")
    return float(np.degrees(np.mean(np.concatenate(errors))))
