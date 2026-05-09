"""Measurement-conditioned proposal models for SO(3)^K particle filters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import PoseSequence
from .measurements import make_synthetic_measurements
from .so3 import left_apply_delta, left_delta, log_map
from .transitions import MLPDeltaTransition, TransitionModel


class ProposalModel:
    """Interface for proposal models used between prediction and weighting."""

    name = "base_proposal"

    @property
    def history_length(self) -> int:
        return 0

    def propose_next(
        self,
        particles: np.ndarray,
        history: list[np.ndarray],
        observation: np.ndarray,
        mask: np.ndarray,
        confidence: np.ndarray,
        joint_noise_sigma_rad: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError


def _clip_tangent_delta(delta: np.ndarray, max_norm_rad: float | None) -> np.ndarray:
    if max_norm_rad is None:
        return delta
    max_norm = float(max_norm_rad)
    if not np.isfinite(max_norm) or max_norm <= 0.0:
        return delta
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norm, 1e-12))
    return delta * scale


def _flat_rotvecs(rotations: np.ndarray, n_rows: int, num_joints: int) -> np.ndarray:
    flat = log_map(rotations).reshape(-1, num_joints * 3)
    if flat.shape[0] == n_rows:
        return flat
    if flat.shape[0] == 1:
        return np.repeat(flat, n_rows, axis=0)
    raise ValueError(f"cannot broadcast rotation feature rows {flat.shape[0]} to {n_rows}")


def _history_delta(history: list[np.ndarray], lag: int, n_rows: int, num_joints: int) -> np.ndarray:
    if len(history) <= lag:
        return np.zeros((n_rows, num_joints * 3), dtype=np.float64)
    delta = left_delta(history[-lag - 1], history[-lag]).reshape(-1, num_joints * 3)
    if delta.shape[0] == n_rows:
        return delta
    if delta.shape[0] == 1:
        return np.repeat(delta, n_rows, axis=0)
    raise ValueError(f"cannot broadcast history feature rows {delta.shape[0]} to {n_rows}")


def _joint_values(values: np.ndarray, n_rows: int, num_joints: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (num_joints,):
        values = np.repeat(values[None, :], n_rows, axis=0)
    elif values.shape == (1, num_joints):
        values = np.repeat(values, n_rows, axis=0)
    elif values.shape != (n_rows, num_joints):
        raise ValueError(f"expected joint values shaped {(num_joints,)} or {(n_rows, num_joints)}, got {values.shape}")
    return values


@dataclass
class MeasurementConditionedMLPProposal(ProposalModel):
    """MLP correction proposal q(x_t | x^-_t, y_t, mask_t, confidence_t)."""

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
    noise_scale: float = 1.0
    max_correction_rad: float | None = None

    name = "measurement_mlp_proposal"

    @property
    def history_length(self) -> int:
        return self.history_steps

    @staticmethod
    def _features(
        predicted: np.ndarray,
        history: list[np.ndarray],
        observation: np.ndarray,
        mask: np.ndarray,
        confidence: np.ndarray,
        joint_noise_sigma_rad: np.ndarray | None,
        history_length: int,
    ) -> np.ndarray:
        predicted = np.asarray(predicted, dtype=np.float64)
        num_joints = predicted.shape[-3]
        n_rows = predicted.reshape(-1, num_joints, 3, 3).shape[0]
        obs = np.asarray(observation, dtype=np.float64)
        mask_values = _joint_values(np.asarray(mask, dtype=np.float64), n_rows, num_joints)
        conf_values = _joint_values(np.asarray(confidence, dtype=np.float64), n_rows, num_joints)
        if joint_noise_sigma_rad is None:
            noise_values = np.zeros((n_rows, num_joints), dtype=np.float64)
        else:
            noise_values = _joint_values(joint_noise_sigma_rad, n_rows, num_joints)
        measurement_delta = left_delta(predicted, obs).reshape(-1, num_joints * 3)
        parts = [
            _flat_rotvecs(predicted, n_rows, num_joints),
            _flat_rotvecs(obs, n_rows, num_joints),
            measurement_delta,
            mask_values,
            conf_values,
            noise_values,
        ]
        for lag in range(1, history_length + 1):
            parts.append(_history_delta(history, lag, n_rows, num_joints))
        return np.concatenate(parts, axis=1)

    @classmethod
    def fit(
        cls,
        sequences: list[PoseSequence],
        transition_model: TransitionModel,
        *,
        history_length: int = 2,
        hidden_dim: int = 128,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        seed: int = 0,
        training_noise_deg: float = 10.0,
        training_occlusion_prob: float = 0.25,
        confidence_noise_std: float = 0.0,
        min_confidence: float = 0.2,
        synthetic_samples_per_sequence: int = 1,
        max_std_rad: float | None = None,
        noise_scale: float = 1.0,
        max_correction_rad: float | None = None,
    ) -> "MeasurementConditionedMLPProposal":
        history_length = max(0, int(history_length))
        rng = np.random.default_rng(seed)
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        num_joints = int(sequences[0].rotations.shape[1])
        for sample_idx in range(max(1, int(synthetic_samples_per_sequence))):
            for seq_idx, seq in enumerate(sequences):
                rotations = np.asarray(seq.rotations, dtype=np.float64)
                if rotations.shape[0] < 2:
                    continue
                meas = make_synthetic_measurements(
                    rotations,
                    training_noise_deg,
                    training_occlusion_prob,
                    np.random.default_rng(seed + 7919 * sample_idx + 101 * seq_idx),
                    confidence_noise_std=confidence_noise_std,
                    min_confidence=min_confidence,
                )
                particle_history: list[np.ndarray] = []
                predicted = rotations[0]
                for t in range(1, rotations.shape[0]):
                    if particle_history:
                        predicted = transition_model.deterministic_next_from_history(particle_history)
                    else:
                        predicted = transition_model.deterministic_next(rotations[t - 1])
                    hist = particle_history or [rotations[t - 1]]
                    features.append(
                        cls._features(
                            predicted,
                            hist,
                            meas.observations[t],
                            meas.mask[t],
                            meas.confidence[t],
                            None if meas.joint_noise_sigma_rad is None else meas.joint_noise_sigma_rad[t],
                            history_length,
                        )[0]
                    )
                    targets.append(left_delta(predicted, rotations[t]).reshape(-1))
                    particle_history.append(rotations[t - 1])
                    particle_history = particle_history[-(history_length + 1) :]
        if not features:
            raise ValueError("need at least one train sequence with two frames")
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        input_mean = np.mean(x, axis=0)
        input_std = np.maximum(np.std(x, axis=0), 1e-6)
        target_mean = np.mean(y, axis=0)
        target_std = np.maximum(np.std(y, axis=0), np.radians(0.25))
        x_train = (x - input_mean) / input_std
        y_train = (y - target_mean) / target_std
        init = np.random.default_rng(seed)
        w1 = init.normal(0.0, np.sqrt(2.0 / max(1, x.shape[1] + hidden_dim)), size=(x.shape[1], hidden_dim))
        b1 = np.zeros(hidden_dim, dtype=np.float64)
        w2 = init.normal(0.0, np.sqrt(2.0 / max(1, hidden_dim + y.shape[1])), size=(hidden_dim, y.shape[1]))
        b2 = np.zeros(y.shape[1], dtype=np.float64)
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
        pred = MLPDeltaTransition._forward_standardized(x_train, w1, b1, w2, b2) * target_std + target_mean
        residual_std = np.maximum(np.std(y - pred, axis=0).reshape(num_joints, 3), np.radians(0.25))
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
            noise_scale=float(noise_scale),
            max_correction_rad=max_correction_rad,
        )

    def _mean_delta(
        self,
        particles: np.ndarray,
        history: list[np.ndarray],
        observation: np.ndarray,
        mask: np.ndarray,
        confidence: np.ndarray,
        joint_noise_sigma_rad: np.ndarray | None,
    ) -> np.ndarray:
        features = self._features(
            particles,
            history,
            observation,
            mask,
            confidence,
            joint_noise_sigma_rad,
            self.history_length,
        )
        standardized = (features - self.input_mean) / self.input_std
        pred = MLPDeltaTransition._forward_standardized(standardized, self.w1, self.b1, self.w2, self.b2)
        delta = pred * self.target_std + self.target_mean
        return delta.reshape(np.asarray(particles).shape[:-2] + (3,))

    def propose_next(
        self,
        particles: np.ndarray,
        history: list[np.ndarray],
        observation: np.ndarray,
        mask: np.ndarray,
        confidence: np.ndarray,
        joint_noise_sigma_rad: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mean = self._mean_delta(particles, history, observation, mask, confidence, joint_noise_sigma_rad)
        if self.noise_scale > 0.0:
            noise = rng.normal(0.0, self.residual_std * float(self.noise_scale), size=mean.shape)
        else:
            noise = np.zeros_like(mean)
        correction = _clip_tangent_delta(mean + noise, self.max_correction_rad)
        return left_apply_delta(correction, particles)

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
            noise_scale=np.asarray(self.noise_scale),
            max_correction_rad=np.asarray(np.nan if self.max_correction_rad is None else self.max_correction_rad),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "MeasurementConditionedMLPProposal":
        with np.load(Path(path), allow_pickle=False) as data:
            max_correction = float(np.asarray(data["max_correction_rad"]).reshape(-1)[0])
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
                noise_scale=float(np.asarray(data["noise_scale"]).reshape(-1)[0]),
                max_correction_rad=None if np.isnan(max_correction) else max_correction,
            )


def build_proposal_model(
    name: str | None,
    train_sequences: list[PoseSequence],
    transition_model: TransitionModel,
    *,
    config: dict[str, Any] | None = None,
) -> ProposalModel | None:
    """Build a proposal model from config, or return ``None`` if disabled."""
    if name is None or str(name).strip().lower() in {"", "none", "off", "disabled"}:
        return None
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized not in {"measurement_mlp", "measurement_conditioned_mlp", "measurement_mlp_proposal"}:
        raise ValueError(f"unknown proposal_model: {name}")
    cfg = config or {}
    checkpoint = cfg.get("measurement_proposal_checkpoint")
    load_checkpoint = bool(cfg.get("proposal_load_checkpoint", bool(checkpoint and Path(checkpoint).exists())))
    save_checkpoint = bool(cfg.get("proposal_save_checkpoint", bool(checkpoint)))
    if checkpoint and load_checkpoint and Path(checkpoint).exists():
        return MeasurementConditionedMLPProposal.load_npz(checkpoint)
    max_std = cfg.get("measurement_proposal_max_std_deg", cfg.get("process_noise_deg"))
    max_corr = cfg.get("measurement_proposal_max_correction_deg")
    model = MeasurementConditionedMLPProposal.fit(
        train_sequences,
        transition_model,
        history_length=int(cfg.get("measurement_proposal_history_length", cfg.get("history_length", 2))),
        hidden_dim=int(cfg.get("measurement_proposal_hidden_dim", 128)),
        epochs=int(cfg.get("measurement_proposal_epochs", 200)),
        learning_rate=float(cfg.get("measurement_proposal_learning_rate", 1e-3)),
        weight_decay=float(cfg.get("measurement_proposal_weight_decay", 1e-4)),
        batch_size=int(cfg.get("measurement_proposal_batch_size", 256)),
        seed=int(cfg.get("measurement_proposal_seed", cfg.get("seed", 0))),
        training_noise_deg=float(cfg.get("measurement_proposal_noise_deg", cfg.get("noise_deg", 10.0))),
        training_occlusion_prob=float(cfg.get("measurement_proposal_occlusion_prob", cfg.get("occlusion_prob", 0.25))),
        confidence_noise_std=float(cfg.get("measurement_proposal_confidence_noise_std", cfg.get("confidence_noise_std", 0.0))),
        min_confidence=float(cfg.get("measurement_proposal_min_confidence", cfg.get("min_confidence", 0.2))),
        synthetic_samples_per_sequence=int(cfg.get("measurement_proposal_synthetic_samples_per_sequence", 1)),
        max_std_rad=None if max_std is None else np.radians(float(max_std)),
        noise_scale=float(cfg.get("measurement_proposal_noise_scale", 1.0)),
        max_correction_rad=None if max_corr is None else np.radians(float(max_corr)),
    )
    if checkpoint and save_checkpoint:
        model.save_npz(checkpoint)
    return model
