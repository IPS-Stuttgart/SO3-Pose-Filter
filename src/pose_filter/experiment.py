"""Config-driven experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import load_dataset, split_sequences
from .evaluation import (
    evaluate_filter,
    robustness_rows,
    trajectory_preview_rows,
    transition_metric_rows,
    write_csv,
    write_json,
)
from .plotting import robustness_plot, trajectory_plot
from .smoothing import SmootherConfig
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
    smoother_config = SmootherConfig(
        ema_alpha=float(config.get("smoother_ema_alpha", 0.35)),
        chordal_window=int(config.get("smoother_chordal_window", 5)),
    )

    transition_rows = transition_metric_rows(
        config["transition_model"],
        model,
        test,
        rollout_horizon=int(config.get("rollout_horizon", 10)),
    )
    filter_rows = evaluate_filter(
        test,
        model,
        float(config["noise_deg"]),
        float(config["occlusion_prob"]),
        int(config["num_particles"]),
        seed,
        proposal_gain=float(config.get("proposal_gain", 0.2)),
        smoother_config=smoother_config,
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
        smoother_config=smoother_config,
    )
    preview_rows = trajectory_preview_rows(
        test[0],
        model,
        float(config["noise_deg"]),
        float(config["occlusion_prob"]),
        int(config["num_particles"]),
        seed + 4242,
        proposal_gain=float(config.get("proposal_gain", 0.2)),
        smoother_config=smoother_config,
    )

    write_csv(output_dir / "transition_metrics.csv", transition_rows)
    write_csv(output_dir / "filter_metrics.csv", filter_rows)
    write_csv(output_dir / "robustness_metrics.csv", robust_rows)
    write_csv(output_dir / "trajectory_preview.csv", preview_rows)
    robustness_plot(output_dir / "plots" / "robustness.svg", robust_rows)
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
        "smoothers": {
            "ema_alpha": smoother_config.ema_alpha,
            "chordal_window": smoother_config.chordal_window,
        },
        "transition_metrics": transition_rows,
        "filter_metrics_mean": {
            "observed_error_deg": float(
                np.nanmean([r["observed_error_deg"] for r in filter_rows])
            ),
            "filter_error_deg": float(
                np.nanmean([r["filter_error_deg"] for r in filter_rows])
            ),
            "persistence_error_deg": float(
                np.nanmean([r["persistence_error_deg"] for r in filter_rows])
            ),
            "smoother_ema_error_deg": float(
                np.nanmean([r["smoother_ema_error_deg"] for r in filter_rows])
            ),
            "smoother_chordal_error_deg": float(
                np.nanmean([r["smoother_chordal_error_deg"] for r in filter_rows])
            ),
            "mean_ess": float(np.nanmean([r["mean_ess"] for r in filter_rows])),
        },
        "outputs": {
            "transition_metrics": str(output_dir / "transition_metrics.csv"),
            "filter_metrics": str(output_dir / "filter_metrics.csv"),
            "robustness_metrics": str(output_dir / "robustness_metrics.csv"),
            "trajectory_preview": str(output_dir / "trajectory_preview.csv"),
            "plots": str(output_dir / "plots"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary
