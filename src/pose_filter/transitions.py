"""Transition baselines for SO(3)^K states."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import PoseSequence, sequence_pairs
from .so3 import geodesic_distance, left_apply_delta, left_delta, log_map


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "gru_delta requires PyTorch. Install the optional torch extra, for example `python -m pip install -e .[torch]`."
        ) from exc
    return torch


def is_torch_available() -> bool:
    try:
        _require_torch()
    except ImportError:
        return False
    return True


def _resolve_torch_device(torch: Any, device: str) -> Any:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gru_delta requested CUDA, but torch.cuda.is_available() is false")
    return torch.device(device)


def _make_gru_delta_module(
    torch: Any,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> Any:
    class _GRUDeltaModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = torch.nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.output = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x: Any) -> Any:
            hidden, _ = self.gru(x)
            return self.output(hidden)

    return _GRUDeltaModule()


def _clip_tangent_delta(delta: np.ndarray, max_norm_rad: float | None) -> np.ndarray:
    if max_norm_rad is None:
        return delta
    max_norm = float(max_norm_rad)
    if not np.isfinite(max_norm) or max_norm <= 0.0:
        return delta
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norm, 1e-12))
    return delta * scale


def _stabilize_tangent_delta(
    delta: np.ndarray,
    *,
    delta_scale: float = 1.0,
    max_norm_rad: float | None = None,
) -> np.ndarray:
    scaled_delta = np.asarray(delta, dtype=np.float64) * delta_scale
    return _clip_tangent_delta(scaled_delta, max_norm_rad)


class TransitionModel:
    name = "base"

    def sample_next(
        self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None
    ) -> np.ndarray:
        raise NotImplementedError

    def deterministic_next(self, x_k: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def log_prob_next(self, x_next: np.ndarray, x_k: np.ndarray) -> np.ndarray | None:
        return None

    def sample_next_from_history(
        self, history: list[np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
        return self.sample_next(history[-1], rng)

    def deterministic_next_from_history(self, history: list[np.ndarray]) -> np.ndarray:
        return self.deterministic_next(history[-1])


class PersistenceTransition(TransitionModel):
    name = "persistence"

    def sample_next(
        self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None
    ) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=np.float64)
        if n_samples is None:
            return x_k.copy()
        return np.repeat(x_k[None, ...], n_samples, axis=0)

    def deterministic_next(self, x_k: np.ndarray) -> np.ndarray:
        return np.asarray(x_k, dtype=np.float64).copy()


@dataclass
class NoisyPersistenceTransition(TransitionModel):
    """Persistence mean with isotropic tangent-space process noise."""

    std_delta: np.ndarray

    name = "noisy_persistence"

    @classmethod
    def from_sequences(
        cls,
        sequences: list[PoseSequence],
        std_rad: float | None = None,
        min_std_rad: float = np.radians(0.25),
        max_std_rad: float | None = None,
    ) -> "NoisyPersistenceTransition":
        if not sequences:
            raise ValueError("need at least one sequence to infer joint shape")
        num_joints = int(sequences[0].rotations.shape[1])
        if std_rad is None:
            if max_std_rad is not None:
                std = float(max_std_rad)
            else:
                std = float(min_std_rad)
        else:
            std = max(float(std_rad), float(min_std_rad))
            if max_std_rad is not None:
                std = min(std, float(max_std_rad))
        return cls(std_delta=np.full((num_joints, 3), std, dtype=np.float64))

    def sample_next(
        self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None
    ) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=np.float64)
        if n_samples is not None:
            base = np.repeat(x_k[None, ...], n_samples, axis=0)
            delta = rng.normal(
                0.0,
                self.std_delta,
                size=(n_samples,) + self.std_delta.shape,
            )
            return left_apply_delta(delta, base)
        delta = rng.normal(0.0, self.std_delta, size=x_k.shape[:-2] + (3,))
        return left_apply_delta(delta, x_k)

    def deterministic_next(self, x_k: np.ndarray) -> np.ndarray:
        return np.asarray(x_k, dtype=np.float64).copy()

    def log_prob_next(self, x_next: np.ndarray, x_k: np.ndarray) -> np.ndarray:
        delta = left_delta(x_k, x_next)
        z = delta / self.std_delta
        return -0.5 * np.sum(
            z * z + np.log(2.0 * np.pi * self.std_delta * self.std_delta),
            axis=(-1, -2),
        )


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

    def _sample_delta(
        self, base_shape: tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        return rng.normal(
            self.mean_delta, self.std_delta, size=base_shape + self.mean_delta.shape
        )

    def sample_next(
        self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None
    ) -> np.ndarray:
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
        return -0.5 * np.sum(
            z * z + np.log(2.0 * np.pi * self.std_delta * self.std_delta), axis=(-1, -2)
        )


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
        residual_std = np.maximum(
            np.std(residual, axis=0).reshape(x.shape[1], 3), min_std_rad
        )
        if max_std_rad is not None:
            residual_std = np.minimum(residual_std, float(max_std_rad))
        return cls(weights=weights, residual_std=residual_std, ridge=ridge)

    def _mean_delta(self, x_k: np.ndarray) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=np.float64)
        features = log_map(x_k).reshape(-1, x_k.shape[-3] * 3)
        design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
        pred = design @ self.weights
        return pred.reshape(x_k.shape[:-2] + (3,))

    def sample_next(
        self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None
    ) -> np.ndarray:
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
        return -0.5 * np.sum(
            z * z + np.log(2.0 * np.pi * self.residual_std * self.residual_std),
            axis=(-1, -2),
        )


@dataclass
class MLPDeltaTransition(TransitionModel):
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    residual_std: np.ndarray
    hidden_dim: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int

    name = "mlp_delta"

    @classmethod
    def fit(
        cls,
        sequences: list[PoseSequence],
        hidden_dim: int = 96,
        epochs: int = 300,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        seed: int = 0,
        min_std_rad: float = np.radians(0.25),
        max_std_rad: float | None = None,
    ) -> "MLPDeltaTransition":
        x, y = sequence_pairs(sequences)
        features = log_map(x).reshape(x.shape[0], -1)
        targets = left_delta(x, y).reshape(x.shape[0], -1)
        input_mean = np.mean(features, axis=0)
        input_std = np.maximum(np.std(features, axis=0), 1e-6)
        target_mean = np.mean(targets, axis=0)
        target_std = np.maximum(np.std(targets, axis=0), min_std_rad)
        x_train = (features - input_mean) / input_std
        y_train = (targets - target_mean) / target_std

        rng = np.random.default_rng(seed)
        n_features = x_train.shape[1]
        n_targets = y_train.shape[1]
        scale1 = np.sqrt(2.0 / max(1, n_features + hidden_dim))
        scale2 = np.sqrt(2.0 / max(1, hidden_dim + n_targets))
        w1: np.ndarray = rng.normal(0.0, scale1, size=(n_features, hidden_dim))
        b1: np.ndarray = np.zeros(hidden_dim, dtype=np.float64)
        w2: np.ndarray = rng.normal(0.0, scale2, size=(hidden_dim, n_targets))
        b2: np.ndarray = np.zeros(n_targets, dtype=np.float64)

        cls._train_adam(
            x_train,
            y_train,
            w1,
            b1,
            w2,
            b2,
            epochs=max(1, int(epochs)),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            batch_size=max(1, int(batch_size)),
            rng=rng,
        )

        pred = cls._forward_standardized(x_train, w1, b1, w2, b2) * target_std + target_mean
        residual = targets - pred
        residual_std = np.maximum(
            np.std(residual, axis=0).reshape(x.shape[1], 3), min_std_rad
        )
        if max_std_rad is not None:
            residual_std = np.minimum(residual_std, float(max_std_rad))
        return cls(
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            residual_std=residual_std,
            hidden_dim=int(hidden_dim),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            seed=int(seed),
        )

    @staticmethod
    def _forward_standardized(
        x: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> np.ndarray:
        hidden = np.tanh(x @ w1 + b1)
        return hidden @ w2 + b2

    @staticmethod
    def _train_adam(
        x_train: np.ndarray,
        y_train: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        rng: np.random.Generator,
    ) -> None:
        params = [w1, b1, w2, b2]
        moments = [np.zeros_like(param) for param in params]
        velocities = [np.zeros_like(param) for param in params]
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        step = 0
        n_samples = x_train.shape[0]
        n_targets = y_train.shape[1]
        for _ in range(epochs):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = order[start : start + batch_size]
                xb = x_train[idx]
                yb = y_train[idx]
                hidden = np.tanh(xb @ w1 + b1)
                pred = hidden @ w2 + b2
                grad_pred = 2.0 * (pred - yb) / float(xb.shape[0] * n_targets)
                grad_w2 = hidden.T @ grad_pred + weight_decay * w2
                grad_b2 = np.sum(grad_pred, axis=0)
                grad_hidden = grad_pred @ w2.T
                grad_z1 = grad_hidden * (1.0 - hidden * hidden)
                grad_w1 = xb.T @ grad_z1 + weight_decay * w1
                grad_b1 = np.sum(grad_z1, axis=0)
                grads = [grad_w1, grad_b1, grad_w2, grad_b2]
                step += 1
                for param, grad, moment, velocity in zip(
                    params, grads, moments, velocities, strict=True
                ):
                    moment *= beta1
                    moment += (1.0 - beta1) * grad
                    velocity *= beta2
                    velocity += (1.0 - beta2) * (grad * grad)
                    moment_hat = moment / (1.0 - beta1**step)
                    velocity_hat = velocity / (1.0 - beta2**step)
                    param -= learning_rate * moment_hat / (
                        np.sqrt(velocity_hat) + eps
                    )

    def _mean_delta(self, x_k: np.ndarray) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=np.float64)
        features = log_map(x_k).reshape(-1, x_k.shape[-3] * 3)
        standardized = (features - self.input_mean) / self.input_std
        pred = self._forward_standardized(
            standardized, self.w1, self.b1, self.w2, self.b2
        )
        delta = pred * self.target_std + self.target_mean
        return delta.reshape(x_k.shape[:-2] + (3,))

    def sample_next(
        self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None
    ) -> np.ndarray:
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
        return -0.5 * np.sum(
            z * z + np.log(2.0 * np.pi * self.residual_std * self.residual_std),
            axis=(-1, -2),
        )

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            residual_std=self.residual_std,
            hidden_dim=np.asarray(self.hidden_dim),
            epochs=np.asarray(self.epochs),
            learning_rate=np.asarray(self.learning_rate),
            weight_decay=np.asarray(self.weight_decay),
            seed=np.asarray(self.seed),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "MLPDeltaTransition":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                input_mean=np.asarray(data["input_mean"], dtype=np.float64),
                input_std=np.asarray(data["input_std"], dtype=np.float64),
                target_mean=np.asarray(data["target_mean"], dtype=np.float64),
                target_std=np.asarray(data["target_std"], dtype=np.float64),
                w1=np.asarray(data["w1"], dtype=np.float64),
                b1=np.asarray(data["b1"], dtype=np.float64),
                w2=np.asarray(data["w2"], dtype=np.float64),
                b2=np.asarray(data["b2"], dtype=np.float64),
                residual_std=np.asarray(data["residual_std"], dtype=np.float64),
                hidden_dim=int(np.asarray(data["hidden_dim"]).reshape(-1)[0]),
                epochs=int(np.asarray(data["epochs"]).reshape(-1)[0]),
                learning_rate=float(np.asarray(data["learning_rate"]).reshape(-1)[0]),
                weight_decay=float(np.asarray(data["weight_decay"]).reshape(-1)[0]),
                seed=int(np.asarray(data["seed"]).reshape(-1)[0]),
            )


@dataclass
class HistoryMLPDeltaTransition(TransitionModel):
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    residual_std: np.ndarray
    history_steps: int
    hidden_dim: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int

    name = "history_mlp_delta"

    @property
    def history_length(self) -> int:
        return self.history_steps

    @classmethod
    def fit(
        cls,
        sequences: list[PoseSequence],
        history_length: int = 2,
        hidden_dim: int = 96,
        epochs: int = 300,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        seed: int = 0,
        min_std_rad: float = np.radians(0.25),
        max_std_rad: float | None = None,
    ) -> "HistoryMLPDeltaTransition":
        history_length = max(1, int(history_length))
        features, targets, num_joints = cls._training_examples(sequences, history_length)
        input_mean = np.mean(features, axis=0)
        input_std = np.maximum(np.std(features, axis=0), 1e-6)
        target_mean = np.mean(targets, axis=0)
        target_std = np.maximum(np.std(targets, axis=0), min_std_rad)
        x_train = (features - input_mean) / input_std
        y_train = (targets - target_mean) / target_std

        rng = np.random.default_rng(seed)
        n_features = x_train.shape[1]
        n_targets = y_train.shape[1]
        scale1 = np.sqrt(2.0 / max(1, n_features + hidden_dim))
        scale2 = np.sqrt(2.0 / max(1, hidden_dim + n_targets))
        w1: np.ndarray = rng.normal(0.0, scale1, size=(n_features, hidden_dim))
        b1: np.ndarray = np.zeros(hidden_dim, dtype=np.float64)
        w2: np.ndarray = rng.normal(0.0, scale2, size=(hidden_dim, n_targets))
        b2: np.ndarray = np.zeros(n_targets, dtype=np.float64)

        MLPDeltaTransition._train_adam(
            x_train,
            y_train,
            w1,
            b1,
            w2,
            b2,
            epochs=max(1, int(epochs)),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            batch_size=max(1, int(batch_size)),
            rng=rng,
        )

        pred = (
            MLPDeltaTransition._forward_standardized(x_train, w1, b1, w2, b2)
            * target_std
            + target_mean
        )
        residual = targets - pred
        residual_std = np.maximum(
            np.std(residual, axis=0).reshape(num_joints, 3), min_std_rad
        )
        if max_std_rad is not None:
            residual_std = np.minimum(residual_std, float(max_std_rad))
        return cls(
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            residual_std=residual_std,
            history_steps=history_length,
            hidden_dim=int(hidden_dim),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            seed=int(seed),
        )

    @staticmethod
    def _training_examples(
        sequences: list[PoseSequence], history_length: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        features = []
        targets = []
        num_joints = sequences[0].rotations.shape[1]
        zero_delta = np.zeros(num_joints * 3, dtype=np.float64)
        for seq in sequences:
            rotations = np.asarray(seq.rotations, dtype=np.float64)
            if rotations.shape[0] < 2:
                continue
            pose_features = log_map(rotations[:-1]).reshape(rotations.shape[0] - 1, -1)
            deltas = left_delta(rotations[:-1], rotations[1:]).reshape(
                rotations.shape[0] - 1, -1
            )
            for t in range(rotations.shape[0] - 1):
                history_parts = []
                for lag in range(1, history_length + 1):
                    history_parts.append(deltas[t - lag] if t - lag >= 0 else zero_delta)
                features.append(np.concatenate([pose_features[t], *history_parts]))
                targets.append(deltas[t])
        if not features:
            raise ValueError("need at least one sequence with two frames")
        return np.asarray(features), np.asarray(targets), num_joints

    def _features_from_history(self, history: list[np.ndarray]) -> np.ndarray:
        current = np.asarray(history[-1], dtype=np.float64)
        num_joints = current.shape[-3]
        pose = log_map(current).reshape(-1, num_joints * 3)
        zeros = np.zeros_like(pose)
        parts = [pose]
        for lag in range(1, self.history_length + 1):
            if len(history) > lag:
                delta = left_delta(history[-lag - 1], history[-lag]).reshape(
                    -1, num_joints * 3
                )
                parts.append(delta)
            else:
                parts.append(zeros)
        return np.concatenate(parts, axis=1)

    def _mean_delta_from_history(self, history: list[np.ndarray]) -> np.ndarray:
        current = np.asarray(history[-1], dtype=np.float64)
        features = self._features_from_history(history)
        standardized = (features - self.input_mean) / self.input_std
        pred = MLPDeltaTransition._forward_standardized(
            standardized, self.w1, self.b1, self.w2, self.b2
        )
        delta = pred * self.target_std + self.target_mean
        return delta.reshape(current.shape[:-2] + (3,))

    def sample_next(
        self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None
    ) -> np.ndarray:
        if n_samples is not None:
            base = np.repeat(np.asarray(x_k, dtype=np.float64)[None, ...], n_samples, axis=0)
            return self.sample_next_from_history([base], rng)
        return self.sample_next_from_history([np.asarray(x_k, dtype=np.float64)], rng)

    def sample_next_from_history(
        self, history: list[np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
        mean = self._mean_delta_from_history(history)
        noise = rng.normal(0.0, self.residual_std, size=mean.shape)
        return left_apply_delta(mean + noise, history[-1])

    def deterministic_next(self, x_k: np.ndarray) -> np.ndarray:
        return self.deterministic_next_from_history([np.asarray(x_k, dtype=np.float64)])

    def deterministic_next_from_history(self, history: list[np.ndarray]) -> np.ndarray:
        return left_apply_delta(self._mean_delta_from_history(history), history[-1])

    def log_prob_next(self, x_next: np.ndarray, x_k: np.ndarray) -> np.ndarray:
        return self.log_prob_next_from_history(x_next, [x_k])

    def log_prob_next_from_history(
        self, x_next: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        delta = left_delta(history[-1], x_next)
        mean = self._mean_delta_from_history(history)
        z = (delta - mean) / self.residual_std
        return -0.5 * np.sum(
            z * z + np.log(2.0 * np.pi * self.residual_std * self.residual_std),
            axis=(-1, -2),
        )

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            residual_std=self.residual_std,
            history_length=np.asarray(self.history_length),
            hidden_dim=np.asarray(self.hidden_dim),
            epochs=np.asarray(self.epochs),
            learning_rate=np.asarray(self.learning_rate),
            weight_decay=np.asarray(self.weight_decay),
            seed=np.asarray(self.seed),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "HistoryMLPDeltaTransition":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                input_mean=np.asarray(data["input_mean"], dtype=np.float64),
                input_std=np.asarray(data["input_std"], dtype=np.float64),
                target_mean=np.asarray(data["target_mean"], dtype=np.float64),
                target_std=np.asarray(data["target_std"], dtype=np.float64),
                w1=np.asarray(data["w1"], dtype=np.float64),
                b1=np.asarray(data["b1"], dtype=np.float64),
                w2=np.asarray(data["w2"], dtype=np.float64),
                b2=np.asarray(data["b2"], dtype=np.float64),
                residual_std=np.asarray(data["residual_std"], dtype=np.float64),
                history_steps=int(np.asarray(data["history_length"]).reshape(-1)[0]),
                hidden_dim=int(np.asarray(data["hidden_dim"]).reshape(-1)[0]),
                epochs=int(np.asarray(data["epochs"]).reshape(-1)[0]),
                learning_rate=float(np.asarray(data["learning_rate"]).reshape(-1)[0]),
                weight_decay=float(np.asarray(data["weight_decay"]).reshape(-1)[0]),
                seed=int(np.asarray(data["seed"]).reshape(-1)[0]),
            )


@dataclass
class GRUDeltaTransition(TransitionModel):
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    residual_std: np.ndarray
    state_dict: dict[str, np.ndarray]
    history_steps: int
    hidden_dim: int
    num_layers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    delta_scale: float = 1.0
    max_delta_norm_rad: float | None = None
    device: str = "cpu"
    _module: Any | None = field(default=None, init=False, repr=False)

    name = "gru_delta"

    @property
    def history_length(self) -> int:
        return self.history_steps

    @classmethod
    def fit(
        cls,
        sequences: list[PoseSequence],
        history_length: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 1,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        device: str = "auto",
        min_std_rad: float = np.radians(0.25),
        max_std_rad: float | None = None,
        delta_scale: float = 1.0,
        max_delta_norm_rad: float | None = None,
    ) -> "GRUDeltaTransition":
        torch = _require_torch()
        torch.manual_seed(int(seed))
        torch_device = _resolve_torch_device(torch, device)
        feature_sequences, target_sequences, num_joints = cls._training_sequences(sequences)
        features = np.concatenate(feature_sequences, axis=0)
        targets = np.concatenate(target_sequences, axis=0)
        input_mean = np.mean(features, axis=0)
        input_std = np.maximum(np.std(features, axis=0), 1e-6)
        target_mean = np.mean(targets, axis=0)
        target_std = np.maximum(np.std(targets, axis=0), min_std_rad)

        x_sequences = [(seq - input_mean) / input_std for seq in feature_sequences]
        y_sequences = [(seq - target_mean) / target_std for seq in target_sequences]
        model = _make_gru_delta_module(
            torch,
            input_dim=features.shape[1],
            hidden_dim=int(hidden_dim),
            output_dim=targets.shape[1],
            num_layers=int(num_layers),
        ).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        rng = np.random.default_rng(seed)
        model.train()
        for _ in range(max(1, int(epochs))):
            for idx in rng.permutation(len(x_sequences)):
                xb = torch.as_tensor(
                    x_sequences[int(idx)][None, ...],
                    dtype=torch.float32,
                    device=torch_device,
                )
                yb = torch.as_tensor(
                    y_sequences[int(idx)][None, ...],
                    dtype=torch.float32,
                    device=torch_device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((model(xb) - yb) ** 2)
                loss.backward()
                optimizer.step()

        predictions = []
        model.eval()
        with torch.no_grad():
            for x_seq in x_sequences:
                xb = torch.as_tensor(
                    x_seq[None, ...],
                    dtype=torch.float32,
                    device=torch_device,
                )
                pred = model(xb)[0].detach().cpu().numpy()
                predictions.append(pred * target_std + target_mean)
        residual = targets - np.concatenate(predictions, axis=0)
        residual_std = np.maximum(
            np.std(residual, axis=0).reshape(num_joints, 3), min_std_rad
        )
        if max_std_rad is not None:
            residual_std = np.minimum(residual_std, float(max_std_rad))

        state_dict = {
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        }
        resolved_device = str(torch_device)
        result = cls(
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
            residual_std=residual_std,
            state_dict=state_dict,
            history_steps=max(1, int(history_length)),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            seed=int(seed),
            delta_scale=float(delta_scale),
            max_delta_norm_rad=max_delta_norm_rad,
            device=resolved_device,
        )
        result._module = model
        return result

    @staticmethod
    def _training_sequences(
        sequences: list[PoseSequence],
    ) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        features = []
        targets = []
        num_joints = sequences[0].rotations.shape[1]
        for seq in sequences:
            rotations = np.asarray(seq.rotations, dtype=np.float64)
            if rotations.shape[0] < 2:
                continue
            features.append(log_map(rotations[:-1]).reshape(rotations.shape[0] - 1, -1))
            targets.append(left_delta(rotations[:-1], rotations[1:]).reshape(rotations.shape[0] - 1, -1))
        if not features:
            raise ValueError("need at least one sequence with two frames")
        return features, targets, num_joints

    def _model(self) -> tuple[Any, Any, Any]:
        torch = _require_torch()
        torch_device = _resolve_torch_device(torch, self.device)
        if self._module is None:
            module = _make_gru_delta_module(
                torch,
                input_dim=int(self.input_mean.shape[0]),
                hidden_dim=int(self.hidden_dim),
                output_dim=int(self.target_mean.shape[0]),
                num_layers=int(self.num_layers),
            ).to(torch_device)
            module.load_state_dict(
                {
                    key: torch.as_tensor(value, dtype=torch.float32, device=torch_device)
                    for key, value in self.state_dict.items()
                }
            )
            module.eval()
            self._module = module
        return self._module, torch, torch_device

    def _features_from_history(self, history: list[np.ndarray]) -> np.ndarray:
        history = history[-(self.history_length + 1) :]
        current = np.asarray(history[-1], dtype=np.float64)
        num_joints = current.shape[-3]
        return np.stack(
            [log_map(np.asarray(entry, dtype=np.float64)).reshape(-1, num_joints * 3) for entry in history],
            axis=1,
        )

    def _mean_delta_from_history(self, history: list[np.ndarray]) -> np.ndarray:
        current = np.asarray(history[-1], dtype=np.float64)
        features = self._features_from_history(history)
        standardized = (features - self.input_mean) / self.input_std
        module, torch, torch_device = self._model()
        with torch.no_grad():
            xb = torch.as_tensor(standardized, dtype=torch.float32, device=torch_device)
            pred = module(xb)[:, -1, :].detach().cpu().numpy()
        delta = pred * self.target_std + self.target_mean
        delta = _stabilize_tangent_delta(
            delta,
            delta_scale=self.delta_scale,
            max_norm_rad=self.max_delta_norm_rad,
        )
        return delta.reshape(current.shape[:-2] + (3,))

    def sample_next(
        self, x_k: np.ndarray, rng: np.random.Generator, n_samples: int | None = None
    ) -> np.ndarray:
        if n_samples is not None:
            base = np.repeat(np.asarray(x_k, dtype=np.float64)[None, ...], n_samples, axis=0)
            return self.sample_next_from_history([base], rng)
        return self.sample_next_from_history([np.asarray(x_k, dtype=np.float64)], rng)

    def sample_next_from_history(
        self, history: list[np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
        mean = self._mean_delta_from_history(history)
        noise = rng.normal(0.0, self.residual_std, size=mean.shape)
        delta = _clip_tangent_delta(mean + noise, self.max_delta_norm_rad)
        return left_apply_delta(delta, history[-1])

    def deterministic_next(self, x_k: np.ndarray) -> np.ndarray:
        return self.deterministic_next_from_history([np.asarray(x_k, dtype=np.float64)])

    def deterministic_next_from_history(self, history: list[np.ndarray]) -> np.ndarray:
        return left_apply_delta(self._mean_delta_from_history(history), history[-1])

    def log_prob_next(self, x_next: np.ndarray, x_k: np.ndarray) -> np.ndarray:
        return self.log_prob_next_from_history(x_next, [x_k])

    def log_prob_next_from_history(
        self, x_next: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        delta = left_delta(history[-1], x_next)
        mean = self._mean_delta_from_history(history)
        z = (delta - mean) / self.residual_std
        return -0.5 * np.sum(
            z * z + np.log(2.0 * np.pi * self.residual_std * self.residual_std),
            axis=(-1, -2),
        )

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "residual_std": self.residual_std,
            "history_length": np.asarray(self.history_length),
            "hidden_dim": np.asarray(self.hidden_dim),
            "num_layers": np.asarray(self.num_layers),
            "epochs": np.asarray(self.epochs),
            "learning_rate": np.asarray(self.learning_rate),
            "weight_decay": np.asarray(self.weight_decay),
            "seed": np.asarray(self.seed),
            "delta_scale": np.asarray(self.delta_scale),
            "max_delta_norm_rad": np.asarray(
                np.nan
                if self.max_delta_norm_rad is None
                else self.max_delta_norm_rad
            ),
            "device": np.asarray(self.device),
            "state_keys": np.asarray(list(self.state_dict), dtype="<U128"),
        }
        for idx, key in enumerate(self.state_dict):
            payload[f"state_{idx}"] = self.state_dict[key]
        np.savez(path, **payload)  # type: ignore[arg-type]

    @classmethod
    def load_npz(cls, path: str | Path) -> "GRUDeltaTransition":
        with np.load(Path(path), allow_pickle=False) as data:
            keys = [str(key) for key in np.asarray(data["state_keys"]).tolist()]
            max_delta_norm_rad = (
                float(np.asarray(data["max_delta_norm_rad"]).reshape(-1)[0])
                if "max_delta_norm_rad" in data
                else float("nan")
            )
            return cls(
                input_mean=np.asarray(data["input_mean"], dtype=np.float64),
                input_std=np.asarray(data["input_std"], dtype=np.float64),
                target_mean=np.asarray(data["target_mean"], dtype=np.float64),
                target_std=np.asarray(data["target_std"], dtype=np.float64),
                residual_std=np.asarray(data["residual_std"], dtype=np.float64),
                state_dict={
                    key: np.asarray(data[f"state_{idx}"], dtype=np.float32)
                    for idx, key in enumerate(keys)
                },
                history_steps=int(np.asarray(data["history_length"]).reshape(-1)[0]),
                hidden_dim=int(np.asarray(data["hidden_dim"]).reshape(-1)[0]),
                num_layers=int(np.asarray(data["num_layers"]).reshape(-1)[0]),
                epochs=int(np.asarray(data["epochs"]).reshape(-1)[0]),
                learning_rate=float(np.asarray(data["learning_rate"]).reshape(-1)[0]),
                weight_decay=float(np.asarray(data["weight_decay"]).reshape(-1)[0]),
                seed=int(np.asarray(data["seed"]).reshape(-1)[0]),
                delta_scale=(
                    float(np.asarray(data["delta_scale"]).reshape(-1)[0])
                    if "delta_scale" in data
                    else 1.0
                ),
                max_delta_norm_rad=(
                    None
                    if np.isnan(max_delta_norm_rad)
                    else max_delta_norm_rad
                ),
                device=str(np.asarray(data["device"]).reshape(-1)[0]),
            )


def build_transition_model(
    name: str,
    train_sequences: list[PoseSequence],
    process_noise_deg: float | None = None,
    config: dict | None = None,
) -> TransitionModel:
    """Fit or construct a transition model by config name."""
    config = {} if config is None else config
    max_std_rad = (
        None if process_noise_deg is None else np.radians(float(process_noise_deg))
    )
    if name == "persistence":
        return PersistenceTransition()
    if name == "deterministic_persistence":
        return PersistenceTransition()
    if name == "noisy_persistence":
        return NoisyPersistenceTransition.from_sequences(
            train_sequences,
            std_rad=(
                None
                if config.get("noisy_persistence_process_noise_deg") is None
                else np.radians(float(config["noisy_persistence_process_noise_deg"]))
            ),
            max_std_rad=max_std_rad,
        )
    if name == "gaussian_rw":
        return GaussianRandomWalkTransition.fit(
            train_sequences, max_std_rad=max_std_rad
        )
    if name == "learned_delta":
        return LearnedDeltaTransition.fit(train_sequences, max_std_rad=max_std_rad)
    if name == "mlp_delta":
        checkpoint = config.get("mlp_transition_checkpoint", config.get("transition_checkpoint"))
        if checkpoint and bool(config.get("transition_load_checkpoint", False)):
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                return MLPDeltaTransition.load_npz(checkpoint_path)
        model = MLPDeltaTransition.fit(
            train_sequences,
            hidden_dim=int(config.get("mlp_hidden_dim", 96)),
            epochs=int(config.get("mlp_epochs", 300)),
            learning_rate=float(config.get("mlp_learning_rate", 1e-3)),
            weight_decay=float(config.get("mlp_weight_decay", 1e-4)),
            batch_size=int(config.get("mlp_batch_size", 256)),
            seed=int(config.get("seed", 0)),
            max_std_rad=max_std_rad,
        )
        if checkpoint and bool(config.get("transition_save_checkpoint", True)):
            model.save_npz(Path(checkpoint))
        return model
    if name == "history_mlp_delta":
        checkpoint = config.get(
            "history_transition_checkpoint", config.get("transition_checkpoint")
        )
        if checkpoint and bool(config.get("transition_load_checkpoint", False)):
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                return HistoryMLPDeltaTransition.load_npz(checkpoint_path)
        history_model = HistoryMLPDeltaTransition.fit(
            train_sequences,
            history_length=int(config.get("history_length", 2)),
            hidden_dim=int(config.get("history_mlp_hidden_dim", config.get("mlp_hidden_dim", 96))),
            epochs=int(config.get("history_mlp_epochs", config.get("mlp_epochs", 300))),
            learning_rate=float(config.get("history_mlp_learning_rate", config.get("mlp_learning_rate", 1e-3))),
            weight_decay=float(config.get("history_mlp_weight_decay", config.get("mlp_weight_decay", 1e-4))),
            batch_size=int(config.get("history_mlp_batch_size", config.get("mlp_batch_size", 256))),
            seed=int(config.get("seed", 0)),
            max_std_rad=max_std_rad,
        )
        if checkpoint and bool(config.get("transition_save_checkpoint", True)):
            history_model.save_npz(Path(checkpoint))
        return history_model
    if name == "gru_delta":
        checkpoint = config.get(
            "gru_transition_checkpoint", config.get("transition_checkpoint")
        )
        if checkpoint and bool(config.get("transition_load_checkpoint", False)):
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                return GRUDeltaTransition.load_npz(checkpoint_path)
        gru_max_delta_deg = config.get("gru_max_delta_deg")
        gru_model = GRUDeltaTransition.fit(
            train_sequences,
            history_length=int(config.get("gru_history_length", config.get("history_length", 8))),
            hidden_dim=int(config.get("gru_hidden_dim", 128)),
            num_layers=int(config.get("gru_num_layers", 1)),
            epochs=int(config.get("gru_epochs", 50)),
            learning_rate=float(config.get("gru_learning_rate", 1e-3)),
            weight_decay=float(config.get("gru_weight_decay", 1e-4)),
            seed=int(config.get("seed", 0)),
            device=str(config.get("gru_device", "auto")),
            max_std_rad=max_std_rad,
            delta_scale=float(config.get("gru_delta_scale", 1.0)),
            max_delta_norm_rad=(
                None
                if gru_max_delta_deg is None
                else np.radians(float(gru_max_delta_deg))
            ),
        )
        if checkpoint and bool(config.get("transition_save_checkpoint", True)):
            gru_model.save_npz(Path(checkpoint))
        return gru_model
    raise ValueError(f"unknown transition_model: {name}")


def one_step_error_deg(model: TransitionModel, sequences: list[PoseSequence]) -> float:
    """Mean one-step deterministic prediction error in degrees."""
    vals = []
    for seq in sequences:
        if seq.rotations.shape[0] < 2:
            continue
        preds = []
        for t in range(seq.rotations.shape[0] - 1):
            start = max(0, t - int(getattr(model, "history_length", 0)))
            history = [seq.rotations[i] for i in range(start, t + 1)]
            preds.append(model.deterministic_next_from_history(history))
        vals.append(geodesic_distance(np.asarray(preds), seq.rotations[1:]).reshape(-1))
    if not vals:
        return float("nan")
    return float(np.degrees(np.mean(np.concatenate(vals))))


def rollout_error_deg(
    model: TransitionModel, sequences: list[PoseSequence], horizon: int
) -> float:
    """Mean deterministic rollout error in degrees over a fixed horizon."""
    errors = []
    horizon = int(horizon)
    for seq in sequences:
        x = seq.rotations[0]
        history = [x]
        max_t = min(horizon, seq.rotations.shape[0] - 1)
        for t in range(1, max_t + 1):
            x = model.deterministic_next_from_history(history)
            errors.append(geodesic_distance(x, seq.rotations[t]).reshape(-1))
            history.append(x)
            keep = int(getattr(model, "history_length", 0)) + 1
            history = history[-keep:]
    if not errors:
        return float("nan")
    return float(np.degrees(np.mean(np.concatenate(errors))))
