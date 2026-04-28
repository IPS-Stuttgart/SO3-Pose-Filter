"""Evaluation helpers for transition models and filtering experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import NamedTuple, TypeVar

import numpy as np

from .data import PoseSequence
from .measurements import make_synthetic_measurements, observed_error_deg
from .particle_filter import run_particle_filter
from .so3 import geodesic_distance, mean_joint_distance_deg
from .transitions import (
    PersistenceTransition,
    TransitionModel,
    one_step_error_deg,
    rollout_error_deg,
)

_T = TypeVar("_T")


class _FilterConfig(NamedTuple):
    num_particles: int
    proposal_gain: float
    factorized_update: bool
    resample_threshold: float


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
    factorized_update: bool = True,
    resample_threshold: float = 0.5,
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
        factorized_update=factorized_update,
        resample_threshold=resample_threshold,
    )
    persistence = PersistenceTransition()
    persistence_estimates_list = [seq.rotations[0]]
    x = seq.rotations[0]
    for _ in range(1, seq.rotations.shape[0]):
        x = persistence.deterministic_next(x)
        persistence_estimates_list.append(x)
    persistence_estimates = np.asarray(persistence_estimates_list)

    return {
        "sequence": seq.name,
        "frames": int(seq.rotations.shape[0]),
        "noise_deg": float(noise_deg),
        "occlusion_prob": float(occlusion_prob),
        "num_particles": int(num_particles),
        "proposal_gain": float(proposal_gain),
        "factorized_update": bool(factorized_update),
        "resample_threshold": float(resample_threshold),
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


def evaluate_filter(
    sequences: list[PoseSequence],
    transition_model: TransitionModel,
    noise_deg: float,
    occlusion_prob: float,
    num_particles: int,
    seed: int,
    proposal_gain: float = 0.2,
    factorized_update: bool = True,
    resample_threshold: float = 0.5,
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
                factorized_update=factorized_update,
                resample_threshold=resample_threshold,
            )
        )
    return rows


def _unique_preserve_order(values: list[_T]) -> list[_T]:
    unique: list[_T] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _mean_row(rows: list[dict]) -> dict:
    return {
        "observed_error_deg": float(
            np.nanmean([r["observed_error_deg"] for r in rows])
        ),
        "filter_error_deg": float(np.nanmean([r["filter_error_deg"] for r in rows])),
        "persistence_error_deg": float(
            np.nanmean([r["persistence_error_deg"] for r in rows])
        ),
        "mean_ess": float(np.nanmean([r["mean_ess"] for r in rows])),
        "mean_resample_count": float(
            np.nanmean([r["resample_count"] for r in rows])
        ),
    }


def ablation_rows(
    sequences: list[PoseSequence],
    transition_model: TransitionModel,
    noise_deg: float,
    occlusion_prob: float,
    base_num_particles: int,
    seed: int,
    base_proposal_gain: float,
    base_factorized_update: bool,
    base_resample_threshold: float,
    particle_counts: list[int],
    proposal_gains: list[float],
    factorized_updates: list[bool],
    resample_thresholds: list[float],
) -> list[dict]:
    """Run one-axis-at-a-time filter ablations around the configured baseline."""

    base = _FilterConfig(
        num_particles=int(base_num_particles),
        proposal_gain=float(base_proposal_gain),
        factorized_update=bool(base_factorized_update),
        resample_threshold=float(base_resample_threshold),
    )
    variants = [("baseline", "baseline", base)]

    for count in _unique_preserve_order(
        [base.num_particles, *[int(x) for x in particle_counts]]
    ):
        cfg = base._replace(num_particles=count)
        variants.append(("num_particles", str(count), cfg))
    for gain in _unique_preserve_order(
        [0.0, base.proposal_gain, *[float(x) for x in proposal_gains]]
    ):
        cfg = base._replace(proposal_gain=gain)
        variants.append(("proposal_gain", f"{gain:g}", cfg))
    for enabled in _unique_preserve_order(
        [False, base.factorized_update, *[bool(x) for x in factorized_updates]]
    ):
        cfg = base._replace(factorized_update=enabled)
        variants.append(("factorized_update", str(enabled).lower(), cfg))
    for threshold in _unique_preserve_order(
        [base.resample_threshold, *[float(x) for x in resample_thresholds]]
    ):
        cfg = base._replace(resample_threshold=threshold)
        variants.append(("resample_threshold", f"{threshold:g}", cfg))

    rows = []
    seen = set()
    for ablation, value, cfg in variants:
        key = (
            ablation,
            value,
            cfg.num_particles,
            cfg.proposal_gain,
            cfg.factorized_update,
            cfg.resample_threshold,
        )
        if key in seen:
            continue
        seen.add(key)
        filter_rows = evaluate_filter(
            sequences,
            transition_model,
            noise_deg,
            occlusion_prob,
            cfg.num_particles,
            seed,
            proposal_gain=cfg.proposal_gain,
            factorized_update=cfg.factorized_update,
            resample_threshold=cfg.resample_threshold,
        )
        rows.append(
            {
                "ablation": ablation,
                "value": value,
                "num_particles": cfg.num_particles,
                "proposal_gain": cfg.proposal_gain,
                "factorized_update": cfg.factorized_update,
                "resample_threshold": cfg.resample_threshold,
                **_mean_row(filter_rows),
            }
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
    factorized_update: bool = True,
    resample_threshold: float = 0.5,
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
                factorized_update=factorized_update,
                resample_threshold=resample_threshold,
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
    factorized_update: bool = True,
    resample_threshold: float = 0.5,
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
        factorized_update=factorized_update,
        resample_threshold=resample_threshold,
    )
    dist_obs = geodesic_distance(seq.rotations, measurements.observations)
    dist_filter = geodesic_distance(seq.rotations, result.estimates)
    rows = []
    for t in range(seq.rotations.shape[0]):
        observed = dist_obs[t][measurements.mask[t]]
        rows.append(
            {
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
        )
    return rows
