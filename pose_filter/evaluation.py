"""Evaluation helpers for transition models and filtering experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import PoseSequence
from .measurements import make_synthetic_measurements, observed_error_deg
from .particle_filter import run_particle_filter
from .so3 import geodesic_distance, left_delta, mean_joint_distance_deg
from .transitions import (
    PersistenceTransition,
    TransitionModel,
    one_step_error_deg,
    rollout_error_deg,
)


FILTER_SUMMARY_KEYS = [
    "observed_error_deg",
    "observed_joint_error_deg",
    "filter_error_deg",
    "persistence_error_deg",
    "filter_observed_joint_error_deg",
    "filter_occluded_joint_error_deg",
    "persistence_observed_joint_error_deg",
    "persistence_occluded_joint_error_deg",
    "filter_acceleration_deg",
    "filter_jerk_deg",
    "filter_acceleration_error_deg",
    "filter_jerk_error_deg",
    "mean_ess",
]


@dataclass(frozen=True)
class FilterEvaluationArtifacts:
    metrics: dict
    per_joint_rows: list[dict]
    temporal_rows: list[dict]


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def _distance_mean_deg(
    truth: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    dist = geodesic_distance(truth, estimate)
    if mask is not None:
        dist = dist[np.asarray(mask, dtype=bool)]
    if dist.size == 0:
        return float("nan")
    return float(np.degrees(np.mean(dist)))


def _per_joint_distance_deg(
    truth: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    dist = geodesic_distance(truth, estimate)
    out = np.full(dist.shape[1], np.nan, dtype=np.float64)
    active = None if mask is None else np.asarray(mask, dtype=bool)
    for joint_idx in range(dist.shape[1]):
        vals = dist[:, joint_idx]
        if active is not None:
            vals = vals[active[:, joint_idx]]
        if vals.size:
            out[joint_idx] = np.degrees(np.mean(vals))
    return out


def _temporal_deltas(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return tangent-space acceleration and jerk for a rotation sequence."""
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.shape[0] < 3:
        empty = np.empty((0,) + rotations.shape[1:-2] + (3,), dtype=np.float64)
        return empty, empty
    velocity = left_delta(rotations[:-1], rotations[1:])
    acceleration = np.diff(velocity, axis=0)
    if acceleration.shape[0] < 2:
        jerk = np.empty((0,) + acceleration.shape[1:], dtype=np.float64)
    else:
        jerk = np.diff(acceleration, axis=0)
    return acceleration, jerk


def _mean_rotvec_norm_deg(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.degrees(np.mean(np.linalg.norm(values, axis=-1))))


def _mean_rotvec_error_deg(values: np.ndarray, truth: np.ndarray) -> float:
    if values.size == 0 or truth.size == 0:
        return float("nan")
    return _mean_rotvec_norm_deg(values - truth)


def temporal_metrics(rotations: np.ndarray, truth: np.ndarray | None = None) -> dict:
    """Summarize temporal acceleration and jerk of an SO(3)^K sequence."""
    acceleration, jerk = _temporal_deltas(rotations)
    row = {
        "acceleration_deg": _mean_rotvec_norm_deg(acceleration),
        "jerk_deg": _mean_rotvec_norm_deg(jerk),
    }
    if truth is not None:
        truth_acceleration, truth_jerk = _temporal_deltas(truth)
        row["acceleration_error_deg"] = _mean_rotvec_error_deg(
            acceleration, truth_acceleration
        )
        row["jerk_error_deg"] = _mean_rotvec_error_deg(jerk, truth_jerk)
    return row


def _temporal_rows(
    sequence: str,
    truth: np.ndarray,
    observations: np.ndarray,
    estimates: np.ndarray,
    persistence_estimates: np.ndarray,
) -> list[dict]:
    rows = []
    for name, rotations in [
        ("truth", truth),
        ("observed", observations),
        ("filter", estimates),
        ("persistence", persistence_estimates),
    ]:
        row = {"sequence": sequence, "estimate": name}
        row.update(temporal_metrics(rotations, truth=truth))
        rows.append(row)
    return rows


def _per_joint_rows(
    sequence: str,
    truth: np.ndarray,
    observations: np.ndarray,
    estimates: np.ndarray,
    persistence_estimates: np.ndarray,
    mask: np.ndarray,
) -> list[dict]:
    observed = _per_joint_distance_deg(truth, observations, mask)
    filter_all = _per_joint_distance_deg(truth, estimates)
    filter_observed = _per_joint_distance_deg(truth, estimates, mask)
    filter_occluded = _per_joint_distance_deg(truth, estimates, ~mask)
    persistence_all = _per_joint_distance_deg(truth, persistence_estimates)
    return [
        {
            "sequence": sequence,
            "joint": joint_idx,
            "observed_error_deg": float(observed[joint_idx]),
            "filter_error_deg": float(filter_all[joint_idx]),
            "filter_observed_joint_error_deg": float(filter_observed[joint_idx]),
            "filter_occluded_joint_error_deg": float(filter_occluded[joint_idx]),
            "persistence_error_deg": float(persistence_all[joint_idx]),
        }
        for joint_idx in range(truth.shape[1])
    ]


def evaluate_filter_sequence_artifacts(
    seq: PoseSequence,
    transition_model: TransitionModel,
    noise_deg: float,
    occlusion_prob: float,
    num_particles: int,
    rng: np.random.Generator,
    proposal_gain: float = 0.2,
) -> dict:
    measurements = make_synthetic_measurements(seq.rotations, noise_deg, occlusion_prob, rng)
    result = run_particle_filter(
        measurements.observations,
        measurements.mask,
        transition_model,
        measurements.noise_sigma_rad,
        num_particles,
        rng,
        proposal_gain=proposal_gain,
    )
    persistence = PersistenceTransition()
    persistence_estimates = [seq.rotations[0]]
    x = seq.rotations[0]
    for _ in range(1, seq.rotations.shape[0]):
        x = persistence.deterministic_next(x)
        persistence_estimates.append(x)
    persistence_estimates = np.asarray(persistence_estimates)

    observed_joint_error = observed_error_deg(
        seq.rotations, measurements.observations, measurements.mask
    )
    metrics = {
        "sequence": seq.name,
        "frames": int(seq.rotations.shape[0]),
        "noise_deg": float(noise_deg),
        "occlusion_prob": float(occlusion_prob),
        "observed_error_deg": observed_joint_error,
        "observed_joint_error_deg": observed_joint_error,
        "filter_error_deg": mean_joint_distance_deg(seq.rotations, result.estimates),
        "persistence_error_deg": mean_joint_distance_deg(seq.rotations, persistence_estimates),
        "filter_observed_joint_error_deg": _distance_mean_deg(
            seq.rotations, result.estimates, measurements.mask
        ),
        "filter_occluded_joint_error_deg": _distance_mean_deg(
            seq.rotations, result.estimates, ~measurements.mask
        ),
        "persistence_observed_joint_error_deg": _distance_mean_deg(
            seq.rotations, persistence_estimates, measurements.mask
        ),
        "persistence_occluded_joint_error_deg": _distance_mean_deg(
            seq.rotations, persistence_estimates, ~measurements.mask
        ),
        "mean_ess": float(np.mean(result.effective_sample_size)),
        "resample_count": int(np.sum(result.resampled)),
    }
    for prefix, rotations in [
        ("observed", measurements.observations),
        ("filter", result.estimates),
        ("persistence", persistence_estimates),
    ]:
        for key, value in temporal_metrics(rotations, truth=seq.rotations).items():
            metrics[f"{prefix}_{key}"] = value

    return FilterEvaluationArtifacts(
        metrics=metrics,
        per_joint_rows=_per_joint_rows(
            seq.name,
            seq.rotations,
            measurements.observations,
            result.estimates,
            persistence_estimates,
            measurements.mask,
        ),
        temporal_rows=_temporal_rows(
            seq.name,
            seq.rotations,
            measurements.observations,
            result.estimates,
            persistence_estimates,
        ),
    )


def evaluate_filter_sequence(
    seq: PoseSequence,
    transition_model: TransitionModel,
    noise_deg: float,
    occlusion_prob: float,
    num_particles: int,
    rng: np.random.Generator,
    proposal_gain: float = 0.2,
) -> dict:
    return evaluate_filter_sequence_artifacts(
        seq,
        transition_model,
        noise_deg,
        occlusion_prob,
        num_particles,
        rng,
        proposal_gain=proposal_gain,
    ).metrics


def evaluate_filter(
    sequences: list[PoseSequence],
    transition_model: TransitionModel,
    noise_deg: float,
    occlusion_prob: float,
    num_particles: int,
    seed: int,
    proposal_gain: float = 0.2,
) -> list[dict]:
    rows = []
    for idx, seq in enumerate(sequences):
        rng = np.random.default_rng(seed + 1009 * idx)
        rows.append(
            evaluate_filter_sequence(
                seq,
                transition_model,
                noise_deg,
                occlusion_prob,
                num_particles,
                rng,
                proposal_gain=proposal_gain,
            )
        )
    return rows


def evaluate_filter_with_artifacts(
    sequences: list[PoseSequence],
    transition_model: TransitionModel,
    noise_deg: float,
    occlusion_prob: float,
    num_particles: int,
    seed: int,
    proposal_gain: float = 0.2,
) -> tuple[list[dict], list[dict], list[dict]]:
    metrics = []
    per_joint_rows = []
    temporal_rows = []
    for idx, seq in enumerate(sequences):
        rng = np.random.default_rng(seed + 1009 * idx)
        artifacts = evaluate_filter_sequence_artifacts(
            seq,
            transition_model,
            noise_deg,
            occlusion_prob,
            num_particles,
            rng,
            proposal_gain=proposal_gain,
        )
        metrics.append(artifacts.metrics)
        per_joint_rows.extend(artifacts.per_joint_rows)
        temporal_rows.extend(artifacts.temporal_rows)
    return metrics, per_joint_rows, temporal_rows


def transition_metric_rows(
    model_name: str,
    model: TransitionModel,
    test_sequences: list[PoseSequence],
    rollout_horizon: int,
) -> list[dict]:
    return [
        {
            "model": model_name,
            "metric": "one_step_error_deg",
            "value": one_step_error_deg(model, test_sequences),
        },
        {
            "model": model_name,
            "metric": "rollout_error_deg",
            "value": rollout_error_deg(model, test_sequences, rollout_horizon),
        },
    ]


def robustness_rows(
    sequences: list[PoseSequence],
    transition_model: TransitionModel,
    noise_grid: list[float],
    occlusion_grid: list[float],
    num_particles: int,
    seed: int,
    proposal_gain: float = 0.2,
) -> list[dict]:
    rows = []
    for noise in noise_grid:
        for occ in occlusion_grid:
            result_rows = evaluate_filter(
                sequences,
                transition_model,
                noise,
                occ,
                num_particles,
                seed + int(noise * 17 + occ * 1000),
                proposal_gain=proposal_gain,
            )
            rows.append(
                {
                    "noise_deg": float(noise),
                    "occlusion_prob": float(occ),
                    "observed_error_deg": _nanmean([r["observed_error_deg"] for r in result_rows]),
                    "observed_joint_error_deg": _nanmean(
                        [r["observed_joint_error_deg"] for r in result_rows]
                    ),
                    "filter_error_deg": _nanmean([r["filter_error_deg"] for r in result_rows]),
                    "persistence_error_deg": _nanmean([r["persistence_error_deg"] for r in result_rows]),
                    "filter_observed_joint_error_deg": _nanmean(
                        [r["filter_observed_joint_error_deg"] for r in result_rows]
                    ),
                    "filter_occluded_joint_error_deg": _nanmean(
                        [r["filter_occluded_joint_error_deg"] for r in result_rows]
                    ),
                    "filter_acceleration_error_deg": _nanmean(
                        [r["filter_acceleration_error_deg"] for r in result_rows]
                    ),
                    "filter_jerk_error_deg": _nanmean(
                        [r["filter_jerk_error_deg"] for r in result_rows]
                    ),
                    "mean_ess": _nanmean([r["mean_ess"] for r in result_rows]),
                }
            )
    return rows


def trajectory_preview_rows(
    seq: PoseSequence,
    transition_model: TransitionModel,
    noise_deg: float,
    occlusion_prob: float,
    num_particles: int,
    seed: int,
    proposal_gain: float = 0.2,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    measurements = make_synthetic_measurements(seq.rotations, noise_deg, occlusion_prob, rng)
    result = run_particle_filter(
        measurements.observations,
        measurements.mask,
        transition_model,
        measurements.noise_sigma_rad,
        num_particles,
        rng,
        proposal_gain=proposal_gain,
    )
    dist_obs = geodesic_distance(seq.rotations, measurements.observations)
    dist_filter = geodesic_distance(seq.rotations, result.estimates)
    rows = []
    for t in range(seq.rotations.shape[0]):
        observed = dist_obs[t][measurements.mask[t]]
        filter_observed = dist_filter[t][measurements.mask[t]]
        filter_occluded = dist_filter[t][~measurements.mask[t]]
        rows.append(
            {
                "frame": t,
                "observed_error_deg": float(np.degrees(np.mean(observed))) if observed.size else float("nan"),
                "filter_error_deg": float(np.degrees(np.mean(dist_filter[t]))),
                "filter_observed_joint_error_deg": float(np.degrees(np.mean(filter_observed)))
                if filter_observed.size
                else float("nan"),
                "filter_occluded_joint_error_deg": float(np.degrees(np.mean(filter_occluded)))
                if filter_occluded.size
                else float("nan"),
                "observed_joint_fraction": float(np.mean(measurements.mask[t])),
                "ess": float(result.effective_sample_size[t]),
            }
        )
    return rows
