"""Evaluation path for real detector or SMPL-fitting measurement outputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .data import PoseSequence, load_dataset, split_sequences
from .detector_import import ImportedMeasurements, load_detector_measurement_dataset
from .evaluation import write_csv, write_json
from .measurements import observed_error_deg
from .model_factory import build_transition_model
from .particle_filter import run_filter
from .so3 import geodesic_distance, mean_joint_distance_deg

REQUIRED_DETECTOR_EVAL_FIELDS = {
    "data_root",
    "measurement_data_root",
    "frame_rate",
    "num_joints",
    "num_particles",
    "transition_model",
}

DETECTOR_SUMMARY_KEYS = [
    "observed_error_deg",
    "observed_joint_error_deg",
    "filter_error_deg",
    "persistence_error_deg",
    "filter_observed_joint_error_deg",
    "filter_occluded_joint_error_deg",
    "persistence_observed_joint_error_deg",
    "persistence_occluded_joint_error_deg",
    "mean_confidence",
    "observed_joint_fraction",
    "mean_ess",
    "min_ess",
    "final_ess",
    "resample_count",
    "resample_fraction",
    "mean_particle_spread_deg",
    "final_particle_spread_deg",
]


def load_detector_eval_config(path: str | Path) -> dict[str, Any]:
    """Load a detector-measurement evaluation JSON config."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    missing = sorted(REQUIRED_DETECTOR_EVAL_FIELDS - set(config))
    if missing:
        raise ValueError(f"detector eval config missing required fields: {', '.join(missing)}")
    return config


def _nanmean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def _distance_mean_deg(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray | None = None) -> float:
    dist = geodesic_distance(truth, estimate)
    if mask is not None:
        dist = dist[np.asarray(mask, dtype=bool)]
    if dist.size == 0:
        return float("nan")
    return float(np.degrees(np.mean(dist)))


def _mean_active_confidence(mask: np.ndarray, confidence: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return float("nan")
    return float(np.mean(np.asarray(confidence, dtype=np.float64)[active]))


def _trim_joint_noise(joint_noise: np.ndarray | None, t_steps: int, mask_shape: tuple[int, int]) -> np.ndarray | None:
    if joint_noise is None:
        return None
    noise = np.asarray(joint_noise, dtype=np.float64)
    if noise.shape == ():
        return noise
    if noise.shape != mask_shape:
        raise ValueError(f"expected joint_noise_sigma_rad shaped {mask_shape}, got {noise.shape}")
    return noise[:t_steps]


def _align_measurement(seq: PoseSequence, measurement: ImportedMeasurements) -> tuple[PoseSequence, ImportedMeasurements]:
    observations = np.asarray(measurement.observations, dtype=np.float64)
    mask = np.asarray(measurement.mask, dtype=bool)
    confidence = np.asarray(measurement.confidence, dtype=np.float64)
    if observations.shape[1:] != seq.rotations.shape[1:]:
        raise ValueError(
            f"measurement shape for {seq.name} has joint/state shape {observations.shape[1:]}, expected {seq.rotations.shape[1:]}"
        )
    if mask.shape != observations.shape[:-2]:
        raise ValueError(f"measurement mask for {seq.name} has shape {mask.shape}, expected {observations.shape[:-2]}")
    if confidence.shape != mask.shape:
        raise ValueError(f"measurement confidence for {seq.name} has shape {confidence.shape}, expected {mask.shape}")
    t_steps = min(seq.rotations.shape[0], observations.shape[0])
    if t_steps < 2:
        raise ValueError(f"need at least two aligned frames for {seq.name}, got {t_steps}")
    aligned_seq = PoseSequence(
        name=seq.name,
        rotations=seq.rotations[:t_steps],
        source_fps=seq.source_fps,
        frame_rate=seq.frame_rate,
    )
    aligned_measurement = replace(
        measurement,
        observations=observations[:t_steps],
        mask=mask[:t_steps],
        confidence=confidence[:t_steps],
        joint_noise_sigma_rad=_trim_joint_noise(measurement.joint_noise_sigma_rad, t_steps, mask.shape),
    )
    return aligned_seq, aligned_measurement


def _persistence_estimates(truth: np.ndarray) -> np.ndarray:
    return np.repeat(truth[0:1], truth.shape[0], axis=0)


def evaluate_detector_measurement_sequence(
    seq: PoseSequence,
    measurement: ImportedMeasurements,
    transition_model,
    num_particles: int,
    rng: np.random.Generator,
    proposal_gain: float = 0.2,
    factorized_update: bool = True,
    resample_threshold: float = 0.5,
    filter_backend: str = "numpy",
) -> dict[str, Any]:
    """Evaluate one ground-truth sequence against imported detector measurements."""
    seq, measurement = _align_measurement(seq, measurement)
    result = run_filter(
        measurement.observations,
        measurement.mask,
        transition_model,
        measurement.noise_sigma_rad,
        int(num_particles),
        rng,
        proposal_gain=proposal_gain,
        confidence=measurement.confidence,
        joint_noise_sigma_rad=measurement.joint_noise_sigma_rad,
        factorized_update=factorized_update,
        resample_threshold=resample_threshold,
        backend=filter_backend,
    )
    persistence_estimates = _persistence_estimates(seq.rotations)
    return {
        "sequence": seq.name,
        "frames": int(seq.rotations.shape[0]),
        "measurement_source": measurement.source,
        "measurement_source_path": measurement.source_path,
        "measurement_noise_sigma_deg": float(np.degrees(measurement.noise_sigma_rad)),
        "observed_joint_fraction": float(np.mean(measurement.mask)),
        "mean_confidence": _mean_active_confidence(measurement.mask, measurement.confidence),
        "num_particles": int(num_particles),
        "proposal_gain": float(proposal_gain),
        "factorized_update": bool(factorized_update),
        "resample_threshold": float(resample_threshold),
        "filter_backend": filter_backend,
        "observed_error_deg": observed_error_deg(
            seq.rotations,
            measurement.observations,
            measurement.mask,
            confidence=measurement.confidence,
        ),
        "observed_joint_error_deg": observed_error_deg(seq.rotations, measurement.observations, measurement.mask),
        "filter_error_deg": mean_joint_distance_deg(seq.rotations, result.estimates),
        "persistence_error_deg": mean_joint_distance_deg(seq.rotations, persistence_estimates),
        "filter_observed_joint_error_deg": _distance_mean_deg(seq.rotations, result.estimates, measurement.mask),
        "filter_occluded_joint_error_deg": _distance_mean_deg(seq.rotations, result.estimates, ~measurement.mask),
        "persistence_observed_joint_error_deg": _distance_mean_deg(seq.rotations, persistence_estimates, measurement.mask),
        "persistence_occluded_joint_error_deg": _distance_mean_deg(seq.rotations, persistence_estimates, ~measurement.mask),
        "mean_ess": float(np.mean(result.effective_sample_size)),
        "min_ess": float(np.min(result.effective_sample_size)),
        "final_ess": float(result.effective_sample_size[-1]),
        "resample_count": int(np.sum(result.resampled)),
        "resample_fraction": float(np.mean(result.resampled)),
        "mean_particle_spread_deg": float(np.mean(result.particle_spread_deg)),
        "final_particle_spread_deg": float(result.particle_spread_deg[-1]),
    }


def run_detector_measurement_eval(config: dict[str, Any]) -> dict[str, Any]:
    """Run an end-to-end filter evaluation using imported detector measurements."""
    seed = int(config.get("seed", 0))
    output_dir = Path(config.get("output_dir", "runs/detector_measurements"))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rate = int(config["frame_rate"])
    num_joints = int(config["num_joints"])
    min_frames = int(config.get("min_frames", 2))

    sequences = load_dataset(
        config["data_root"],
        config.get("dataset_subset", ""),
        frame_rate,
        num_joints,
        max_sequences=config.get("max_sequences"),
        min_frames=min_frames,
    )
    train, val, test = split_sequences(
        sequences,
        train_fraction=float(config.get("train_fraction", 0.7)),
        val_fraction=float(config.get("val_fraction", 0.15)),
        seed=seed,
    )
    if not test:
        test = val or train

    measurement_noise_deg = float(config.get("measurement_noise_deg", config.get("noise_deg", 10.0)))
    measurements = load_detector_measurement_dataset(
        config["measurement_data_root"],
        config.get("measurement_dataset_subset", config.get("dataset_subset", "")),
        frame_rate,
        num_joints,
        measurement_noise_deg,
        max_sequences=config.get("measurement_max_sequences"),
        min_frames=min_frames,
        pose_key=config.get("measurement_pose_key"),
        mask_key=config.get("measurement_mask_key"),
        confidence_key=config.get("measurement_confidence_key"),
        joint_noise_key=config.get("measurement_joint_noise_key"),
        confidence_scale=float(config.get("measurement_confidence_scale", 1.0)),
        quaternion_order=str(config.get("measurement_quaternion_order", "xyzw")),
    )

    model = build_transition_model(
        config["transition_model"],
        train,
        process_noise_deg=config.get("process_noise_deg"),
        config=config,
    )
    allow_missing = bool(config.get("allow_missing_measurements", False))
    missing = []
    rows = []
    for idx, seq in enumerate(test):
        measurement = measurements.get(seq.name)
        if measurement is None:
            missing.append(seq.name)
            if allow_missing:
                continue
            raise ValueError(f"no detector measurement found for test sequence '{seq.name}'")
        rows.append(
            evaluate_detector_measurement_sequence(
                seq,
                measurement,
                model,
                int(config["num_particles"]),
                np.random.default_rng(seed + 1009 * idx),
                proposal_gain=float(config.get("proposal_gain", 0.2)),
                factorized_update=bool(config.get("factorized_update", True)),
                resample_threshold=float(config.get("resample_threshold", 0.5)),
                filter_backend=str(config.get("filter_backend", "numpy")),
            )
        )
    if not rows:
        raise ValueError("no detector measurement rows were evaluated")

    write_csv(output_dir / "detector_filter_metrics.csv", rows)
    summary = {
        "transition_model": config["transition_model"],
        "measurement_data_root": str(config["measurement_data_root"]),
        "measurement_count": len(measurements),
        "missing_measurements": missing,
        "row_count": len(rows),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "frame_rate": frame_rate,
        "num_joints": num_joints,
        "num_particles": int(config["num_particles"]),
        "filter_backend": str(config.get("filter_backend", "numpy")),
        "means": {key: _nanmean([float(row[key]) for row in rows]) for key in DETECTOR_SUMMARY_KEYS},
        "outputs": {
            "detector_filter_metrics": str(output_dir / "detector_filter_metrics.csv"),
            "summary": str(output_dir / "detector_measurement_eval_summary.json"),
        },
    }
    write_json(output_dir / "detector_measurement_eval_summary.json", summary)
    return summary
