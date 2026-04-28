"""Evaluation helpers for transition models and filtering experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import PoseSequence
from .measurements import make_synthetic_measurements, observed_error_deg
from .particle_filter import run_particle_filter
from .smoothing import SmootherConfig, run_baseline_smoothers
from .so3 import geodesic_distance, mean_joint_distance_deg
from .transitions import (
    PersistenceTransition,
    TransitionModel,
    one_step_error_deg,
    rollout_error_deg,
)


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


def evaluate_filter_sequence(
    seq: PoseSequence,
    transition_model: TransitionModel,
    noise_deg: float,
    occlusion_prob: float,
    num_particles: int,
    rng: np.random.Generator,
    proposal_gain: float = 0.2,
    smoother_config: SmootherConfig | None = None,
) -> dict:
    measurements = make_synthetic_measurements(
        seq.rotations, noise_deg, occlusion_prob, rng
    )
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
    persistence_estimates_list = [seq.rotations[0]]
    x = seq.rotations[0]
    for _ in range(1, seq.rotations.shape[0]):
        x = persistence.deterministic_next(x)
        persistence_estimates_list.append(x)
    persistence_estimates = np.asarray(persistence_estimates_list)
    smoother_estimates = run_baseline_smoothers(
        measurements.observations, measurements.mask, smoother_config
    )

    row = {
        "sequence": seq.name,
        "frames": int(seq.rotations.shape[0]),
        "noise_deg": float(noise_deg),
        "occlusion_prob": float(occlusion_prob),
        "observed_error_deg": observed_error_deg(
            seq.rotations, measurements.observations, measurements.mask
        ),
        "filter_error_deg": mean_joint_distance_deg(seq.rotations, result.estimates),
        "persistence_error_deg": mean_joint_distance_deg(
            seq.rotations, persistence_estimates
        ),
        "mean_ess": float(np.mean(result.effective_sample_size)),
        "resample_count": int(np.sum(result.resampled)),
    }
    for name, estimates in smoother_estimates.items():
        row[f"{name}_error_deg"] = mean_joint_distance_deg(seq.rotations, estimates)
    return row


def evaluate_filter(
    sequences: list[PoseSequence],
    transition_model: TransitionModel,
    noise_deg: float,
    occlusion_prob: float,
    num_particles: int,
    seed: int,
    proposal_gain: float = 0.2,
    smoother_config: SmootherConfig | None = None,
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
                smoother_config=smoother_config,
            )
        )
    return rows


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
    smoother_config: SmootherConfig | None = None,
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
                smoother_config=smoother_config,
            )
            rows.append(
                {
                    "noise_deg": float(noise),
                    "occlusion_prob": float(occ),
                    "observed_error_deg": float(
                        np.nanmean([r["observed_error_deg"] for r in result_rows])
                    ),
                    "filter_error_deg": float(
                        np.nanmean([r["filter_error_deg"] for r in result_rows])
                    ),
                    "persistence_error_deg": float(
                        np.nanmean([r["persistence_error_deg"] for r in result_rows])
                    ),
                    "smoother_ema_error_deg": float(
                        np.nanmean([r["smoother_ema_error_deg"] for r in result_rows])
                    ),
                    "smoother_chordal_error_deg": float(
                        np.nanmean(
                            [r["smoother_chordal_error_deg"] for r in result_rows]
                        )
                    ),
                    "mean_ess": float(np.nanmean([r["mean_ess"] for r in result_rows])),
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
    smoother_config: SmootherConfig | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    measurements = make_synthetic_measurements(
        seq.rotations, noise_deg, occlusion_prob, rng
    )
    result = run_particle_filter(
        measurements.observations,
        measurements.mask,
        transition_model,
        measurements.noise_sigma_rad,
        num_particles,
        rng,
        proposal_gain=proposal_gain,
    )
    smoother_estimates = run_baseline_smoothers(
        measurements.observations, measurements.mask, smoother_config
    )
    dist_obs = geodesic_distance(seq.rotations, measurements.observations)
    dist_filter = geodesic_distance(seq.rotations, result.estimates)
    dist_smoothers = {
        name: geodesic_distance(seq.rotations, estimates)
        for name, estimates in smoother_estimates.items()
    }
    rows = []
    for t in range(seq.rotations.shape[0]):
        observed = dist_obs[t][measurements.mask[t]]
        row = {
            "frame": t,
            "observed_error_deg": (
                float(np.degrees(np.mean(observed)))
                if observed.size
                else float("nan")
            ),
            "filter_error_deg": float(np.degrees(np.mean(dist_filter[t]))),
            "observed_joint_fraction": float(np.mean(measurements.mask[t])),
            "ess": float(result.effective_sample_size[t]),
        }
        for name, dist in dist_smoothers.items():
            row[f"{name}_error_deg"] = float(np.degrees(np.mean(dist[t])))
        rows.append(row)
    return rows
