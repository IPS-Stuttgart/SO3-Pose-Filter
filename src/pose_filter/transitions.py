"""Transition baselines for SO(3)^K states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import PoseSequence, sequence_pairs
from .so3 import geodesic_distance, left_apply_delta, left_delta, log_map


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
    if name == "gaussian_rw":
        return GaussianRandomWalkTransition.fit(
            train_sequences, max_std_rad=max_std_rad
        )
    if name == "learned_delta":
        return LearnedDeltaTransition.fit(train_sequences, max_std_rad=max_std_rad)
    if name == "mlp_delta":
        checkpoint = config.get("transition_checkpoint")
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


def rollout_error_deg(
    model: TransitionModel, sequences: list[PoseSequence], horizon: int
) -> float:
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
