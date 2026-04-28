"""Config-driven experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import load_dataset, split_sequences
from .evaluation import (
    FILTER_SUMMARY_KEYS,
    evaluate_filter_with_artifacts,
    robustness_rows,
    trajectory_preview_rows,
    transition_metric_rows,
    write_csv,
    write_json,
)
from .plotting import robustness_plot, trajectory_plot
from .transitions import build_transition_model

REQUIRED_CONFIG_FIELDS = {
    "data_root",
    "dataset_subset",
    "frame_rate",
    "num_joints",
    "noise_deg",
    "occlusion_prob",
    "num_particles",
    "transition_model",
}


def _mean_metric(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"config missing required fields: {', '.join(missing)}")
    return config


def run_experiment(config: dict) -> dict:
    seed = int(config.get("seed", 0))
    output_dir = Path(config.get("output_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_dataset(
        config["data_root"],
        config.get("dataset_subset", ""),
        int(config["frame_rate"]),
        int(config["num_joints"]),
        max_sequences=config.get("max_sequences"),
        min_frames=int(config.get("min_frames", 2)),
    )
    train, val, test = split_sequences(
        sequences,
        train_fraction=float(config.get("train_fraction", 0.7)),
        val_fraction=float(config.get("val_fraction", 0.15)),
        seed=seed,
    )
    if not test:
        test = val or train

    model = build_transition_model(
        config["transition_model"],
        train,
        process_noise_deg=config.get("process_noise_deg"),
    )

    transition_rows = transition_metric_rows(
        config["transition_model"],
        model,
        test,
        rollout_horizon=int(config.get("rollout_horizon", 10)),
    )
    filter_rows, per_joint_rows, temporal_rows = evaluate_filter_with_artifacts(
        test,
        model,
        float(config["noise_deg"]),
        float(config["occlusion_prob"]),
        int(config["num_particles"]),
        seed,
        proposal_gain=float(config.get("proposal_gain", 0.2)),
    )
    robust_rows = robustness_rows(
        test,
        model,
        [float(x) for x in config.get("robustness_noise_deg", [config["noise_deg"]])],
        [
            float(x)
            for x in config.get("robustness_occlusion_prob", [config["occlusion_prob"]])
        ],
        int(config["num_particles"]),
        seed,
        proposal_gain=float(config.get("proposal_gain", 0.2)),
    )
    preview_rows = trajectory_preview_rows(
        test[0],
        model,
        float(config["noise_deg"]),
        float(config["occlusion_prob"]),
        int(config["num_particles"]),
        seed + 4242,
        proposal_gain=float(config.get("proposal_gain", 0.2)),
    )

    write_csv(output_dir / "transition_metrics.csv", transition_rows)
    write_csv(output_dir / "filter_metrics.csv", filter_rows)
    write_csv(output_dir / "per_joint_metrics.csv", per_joint_rows)
    write_csv(output_dir / "temporal_metrics.csv", temporal_rows)
    write_csv(output_dir / "robustness_metrics.csv", robust_rows)
    write_csv(output_dir / "trajectory_preview.csv", preview_rows)
    robustness_plot(output_dir / "plots" / "robustness.svg", robust_rows)
    robustness_plot(
        output_dir / "plots" / "robustness_occluded.svg",
        robust_rows,
        metric="filter_occluded_joint_error_deg",
        title="Occluded-Joint Robustness",
        y_label="occluded-joint filter error (deg)",
    )
    robustness_plot(
        output_dir / "plots" / "robustness_acceleration.svg",
        robust_rows,
        metric="filter_acceleration_error_deg",
        title="Temporal Acceleration Robustness",
        y_label="acceleration error (deg)",
    )
    trajectory_plot(output_dir / "plots" / "trajectory_preview.svg", preview_rows)

    summary = {
        "transition_model": config["transition_model"],
        "num_sequences": len(sequences),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "frame_rate": int(config["frame_rate"]),
        "num_joints": int(config["num_joints"]),
        "noise_deg": float(config["noise_deg"]),
        "occlusion_prob": float(config["occlusion_prob"]),
        "num_particles": int(config["num_particles"]),
        "process_noise_deg": config.get("process_noise_deg"),
        "proposal_gain": float(config.get("proposal_gain", 0.2)),
        "transition_metrics": transition_rows,
        "filter_metrics_mean": {
            key: _mean_metric(filter_rows, key) for key in FILTER_SUMMARY_KEYS
        },
        "outputs": {
            "transition_metrics": str(output_dir / "transition_metrics.csv"),
            "filter_metrics": str(output_dir / "filter_metrics.csv"),
            "per_joint_metrics": str(output_dir / "per_joint_metrics.csv"),
            "temporal_metrics": str(output_dir / "temporal_metrics.csv"),
            "robustness_metrics": str(output_dir / "robustness_metrics.csv"),
            "trajectory_preview": str(output_dir / "trajectory_preview.csv"),
            "plots": str(output_dir / "plots"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary
