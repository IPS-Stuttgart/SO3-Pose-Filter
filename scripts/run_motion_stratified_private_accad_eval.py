from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_filter.data import load_amass_sequence  # noqa: E402
from pose_filter.evaluation import write_csv, write_json  # noqa: E402
from prepare_amass_windows import prepare_windows  # noqa: E402
from run_first_results_benchmark import run_first_results_benchmark  # noqa: E402

DEFAULT_METHODS = (
    "raw",
    "persistence",
    "deterministic_persistence_pf",
    "noisy_persistence_pf",
    "constant_velocity",
    "gaussian_rw",
    "mlp_delta",
    "history_mlp_delta",
    "gru_delta",
)

MOTION_BINS = (
    ("low_motion", 0.0, 0.5),
    ("medium_motion", 0.5, 1.5),
    ("high_motion", 1.5, float("inf")),
)


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def load_motion_stratified_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def _float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


def _sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _motion_bin(motion_deg_per_frame: float) -> str:
    motion = float(motion_deg_per_frame)
    for name, lower, upper in MOTION_BINS:
        if lower <= motion < upper:
            return name
    return MOTION_BINS[-1][0]


def _motion_bin_bounds(name: str) -> tuple[float, float]:
    for bin_name, lower, upper in MOTION_BINS:
        if bin_name == name:
            return float(lower), float(upper)
    raise ValueError(f"unknown motion bin: {name}")


def _group_windows_by_motion_bin(window_report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in MOTION_BINS}
    for row in window_report.get("selected", []):
        bin_name = _motion_bin(float(row["motion_deg_per_frame"]))
        grouped[bin_name].append(row)
    return {key: value for key, value in grouped.items() if value}


def _safe_name_component(value: str, limit: int = 96) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return safe[:limit].strip("_.-") or "segment"


def _source_slug(row: dict[str, Any], fallback: Path) -> str:
    source_path = Path(str(row.get("source_path") or fallback))
    informative_parts = source_path.parts[-3:]
    if len(informative_parts) >= 2:
        stem = "_".join(
            Path(part).stem if idx == len(informative_parts) - 1 else part
            for idx, part in enumerate(informative_parts)
        )
    else:
        stem = fallback.stem
    return _safe_name_component(stem)


def _copy_motion_bin_segments(
    windows: list[dict[str, Any]],
    bin_name: str,
    output_root: Path,
) -> Path:
    bin_dir = output_root / "motion_bin_segments" / bin_name
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(windows):
        source = Path(str(row["output_path"]))
        target = bin_dir / f"segment_{idx:03d}_{_source_slug(row, source)}.npz"
        shutil.copy2(source, target)
    return bin_dir


def _segment_diagnostics(
    bin_name: str,
    bin_dir: Path,
    windows: list[dict[str, Any]],
    *,
    frame_rate: int,
    num_joints: int,
    min_frames: int,
    max_examples: int = 5,
) -> dict[str, Any]:
    files = sorted(bin_dir.rglob("*.npz"))
    usable = 0
    rejection_examples = []
    for path in files:
        try:
            seq = load_amass_sequence(path, frame_rate=frame_rate, num_joints=num_joints)
            frames = int(seq.rotations.shape[0])
            if frames < int(min_frames):
                if len(rejection_examples) < max_examples:
                    rejection_examples.append(
                        {
                            "file": str(path.relative_to(bin_dir)),
                            "reason": f"{frames} frames after downsampling, need at least {int(min_frames)}",
                        }
                    )
                continue
            usable += 1
        except Exception as exc:
            if len(rejection_examples) < max_examples:
                rejection_examples.append(
                    {"file": str(path.relative_to(bin_dir)), "reason": str(exc)}
                )
    source_examples = []
    for row in windows[:max_examples]:
        source = Path(str(row.get("source_path") or row.get("output_path", "")))
        source_examples.append("/".join(source.parts[-3:]))
    return {
        "motion_bin": bin_name,
        "selected_window_count": len(windows),
        "segment_dir": str(bin_dir),
        "npz_count": len(files),
        "usable_count": usable,
        "min_frames": int(min_frames),
        "frame_rate": int(frame_rate),
        "num_joints": int(num_joints),
        "source_examples": source_examples,
        "rejection_examples": rejection_examples,
    }


def _validate_motion_bin_segments(
    grouped_windows: dict[str, list[dict[str, Any]]],
    bin_data_dirs: dict[str, Path],
    diagnostics_path: Path,
    *,
    frame_rate: int,
    num_joints: int,
    min_frames: int,
) -> dict[str, Any]:
    diagnostics = {
        bin_name: _segment_diagnostics(
            bin_name,
            bin_data_dirs[bin_name],
            windows,
            frame_rate=frame_rate,
            num_joints=num_joints,
            min_frames=min_frames,
        )
        for bin_name, windows in grouped_windows.items()
    }
    write_json(diagnostics_path, diagnostics)
    bad_bins = {
        bin_name: details
        for bin_name, details in diagnostics.items()
        if int(details["usable_count"]) <= 0
    }
    if bad_bins:
        compact = {
            bin_name: {
                "selected_window_count": details["selected_window_count"],
                "npz_count": details["npz_count"],
                "usable_count": details["usable_count"],
                "source_examples": details["source_examples"],
                "rejection_examples": details["rejection_examples"],
            }
            for bin_name, details in bad_bins.items()
        }
        raise ValueError(
            "one or more motion bins contain no usable AMASS segment files after materialization; "
            f"see {diagnostics_path}\n{json.dumps(compact, indent=2)}"
        )
    return diagnostics


def _copy_row_with_run(
    row: dict[str, str],
    run: dict[str, Any],
    bin_name: str,
) -> dict[str, Any]:
    lower, upper = _motion_bin_bounds(bin_name)
    out: dict[str, Any] = {
        "motion_bin": bin_name,
        "motion_bin_min_deg_per_frame": lower,
        "motion_bin_max_deg_per_frame": upper,
        "seed": int(run["seed"]),
        "num_particles": int(run["num_particles"]),
        "run_dir": str(run["output_dir"]),
    }
    out.update(row)
    return out


def _aggregate_method_means(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups = sorted({tuple(str(row[key]) for key in group_keys) for row in rows})
    out = []
    for group in groups:
        group_rows = [
            row for row in rows if tuple(str(row[key]) for key in group_keys) == group
        ]
        record: dict[str, Any] = dict(zip(group_keys, group, strict=True))
        tracking = [float(row["tracking_error_deg"]) for row in group_rows]
        raw_improvement = [float(row["improvement_vs_raw_deg"]) for row in group_rows]
        persistence_improvement = [
            float(row["improvement_vs_persistence_deg"]) for row in group_rows
        ]
        record.update(
            {
                "mean_tracking_error_deg": _mean(tracking),
                "std_tracking_error_deg": _std(tracking),
                "sem_tracking_error_deg": _sem(tracking),
                "mean_improvement_vs_raw_deg": _mean(raw_improvement),
                "std_improvement_vs_raw_deg": _std(raw_improvement),
                "sem_improvement_vs_raw_deg": _sem(raw_improvement),
                "mean_improvement_vs_persistence_deg": _mean(
                    persistence_improvement
                ),
                "std_improvement_vs_persistence_deg": _std(
                    persistence_improvement
                ),
                "sem_improvement_vs_persistence_deg": _sem(
                    persistence_improvement
                ),
                "row_count": len(group_rows),
            }
        )
        out.append(record)
    return out


def _aggregate_transition_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted(
        {
            (str(row["motion_bin"]), str(row["model"]), str(row["metric"]))
            for row in rows
        }
    )
    out = []
    for motion_bin, model, metric in groups:
        values = [
            float(row["value"])
            for row in rows
            if str(row["motion_bin"]) == motion_bin
            and str(row["model"]) == model
            and str(row["metric"]) == metric
        ]
        out.append(
            {
                "motion_bin": motion_bin,
                "model": model,
                "metric": metric,
                "mean_value": _mean(values),
                "std_value": _std(values),
                "sem_value": _sem(values),
                "row_count": len(values),
            }
        )
    return out


def _robustness_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({(str(row["motion_bin"]), str(row["method"])) for row in rows})
    out = []
    for motion_bin, method in groups:
        values = [
            float(row["mean_tracking_error_deg"])
            for row in rows
            if str(row["motion_bin"]) == motion_bin and str(row["method"]) == method
        ]
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        out.append(
            {
                "motion_bin": motion_bin,
                "method": method,
                "best_tracking_error_deg": float(np.min(finite))
                if finite.size
                else float("nan"),
                "median_tracking_error_deg": float(np.median(finite))
                if finite.size
                else float("nan"),
                "worst_tracking_error_deg": float(np.max(finite))
                if finite.size
                else float("nan"),
                "mean_tracking_error_deg": _mean(values),
                "grid_point_count": len(values),
            }
        )
    return out


def _particle_collapse_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({(str(row["motion_bin"]), str(row["method"])) for row in rows})
    out = []
    for motion_bin, method in groups:
        method_rows = [
            row
            for row in rows
            if str(row["motion_bin"]) == motion_bin
            and str(row["method"]) == method
            and str(row.get("filter_backend", "none")) != "none"
        ]
        if not method_rows:
            continue
        out.append(
            {
                "motion_bin": motion_bin,
                "method": method,
                "mean_ess": _mean([float(row["mean_ess"]) for row in method_rows]),
                "min_ess": _mean([float(row["min_ess"]) for row in method_rows]),
                "final_ess": _mean([float(row["final_ess"]) for row in method_rows]),
                "resample_fraction": _mean(
                    [float(row["resample_fraction"]) for row in method_rows]
                ),
                "mean_particle_spread_deg": _mean(
                    [float(row["mean_particle_spread_deg"]) for row in method_rows]
                ),
                "min_particle_spread_deg": _mean(
                    [float(row["min_particle_spread_deg"]) for row in method_rows]
                ),
                "final_particle_spread_deg": _mean(
                    [float(row["final_particle_spread_deg"]) for row in method_rows]
                ),
                "collapse_fraction": _mean(
                    [float(row["collapse_fraction"]) for row in method_rows]
                ),
                "filter_reappeared_joint_error_deg": _mean(
                    [
                        float(row["filter_reappeared_joint_error_deg"])
                        for row in method_rows
                    ]
                ),
                "persistence_reappeared_joint_error_deg": _mean(
                    [
                        float(row["persistence_reappeared_joint_error_deg"])
                        for row in method_rows
                    ]
                ),
                "reappeared_joint_count": _mean(
                    [float(row["reappeared_joint_count"]) for row in method_rows]
                ),
                "row_count": len(method_rows),
            }
        )
    return out


def _transition_model_for_method(method: str) -> str:
    return {
        "deterministic_persistence_pf": "deterministic_persistence",
        "noisy_persistence_pf": "noisy_persistence",
        "pyrecest_pf": "gaussian_rw",
    }.get(method, method)


def _transition_tracking_diagnostics(
    method_means_by_motion: list[dict[str, Any]],
    transition_means: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    transition_lookup = {
        (row["motion_bin"], row["model"], row["metric"]): row
        for row in transition_means
    }
    out = []
    for row in method_means_by_motion:
        method = str(row["method"])
        if method in {"raw", "persistence"}:
            continue
        transition_model = _transition_model_for_method(method)
        one_step = transition_lookup.get(
            (str(row["motion_bin"]), transition_model, "one_step_error_deg")
        )
        rollout = transition_lookup.get(
            (str(row["motion_bin"]), transition_model, "rollout_error_deg")
        )
        if one_step is None and rollout is None:
            continue
        out.append(
            {
                "motion_bin": row["motion_bin"],
                "method": method,
                "transition_model": transition_model,
                "mean_tracking_error_deg": row["mean_tracking_error_deg"],
                "sem_tracking_error_deg": row["sem_tracking_error_deg"],
                "mean_one_step_error_deg": one_step["mean_value"]
                if one_step
                else float("nan"),
                "mean_rollout_error_deg": rollout["mean_value"]
                if rollout
                else float("nan"),
                "tracking_row_count": row["row_count"],
                "transition_row_count": max(
                    int(one_step["row_count"]) if one_step else 0,
                    int(rollout["row_count"]) if rollout else 0,
                ),
            }
        )
    return out


def _benchmark_config(
    config: dict[str, Any],
    bin_data_dir: Path,
    output_dir: Path,
    seed: int,
    num_particles: int,
) -> dict[str, Any]:
    run_config = dict(config)
    run_config.update(
        {
            "data_root": str(bin_data_dir),
            "dataset_subset": "",
            "seed": int(seed),
            "num_particles": int(num_particles),
            "output_dir": str(output_dir / "base"),
            "benchmark_output_dir": str(output_dir),
        }
    )
    checkpoint_dir = output_dir / "checkpoints"
    run_config.setdefault(
        "mlp_transition_checkpoint",
        str(checkpoint_dir / "mlp_delta_checkpoint.npz"),
    )
    run_config.setdefault(
        "history_transition_checkpoint",
        str(checkpoint_dir / "history_mlp_delta_checkpoint.npz"),
    )
    run_config.setdefault(
        "gru_transition_checkpoint",
        str(checkpoint_dir / "gru_delta_checkpoint.npz"),
    )
    return run_config


def _write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = str(summary.get("dataset_name", "ACCAD"))
    lines = [
        f"# Motion-Stratified Private {dataset_name} Evaluation",
        "",
        "Generated outputs are local artifacts and should stay under ignored output folders.",
        "",
        f"- source data root: `{summary['source_data_root']}`",
        f"- selected windows: `{summary['window_report']['selected_count']}`",
        f"- benchmark runs: `{len(summary['runs'])}`",
        "",
        "## Motion-bin counts",
        "",
        "| motion bin | windows |",
        "| --- | ---: |",
    ]
    for bin_name, count in summary["motion_bin_counts"].items():
        lines.append(f"| {bin_name} | {count} |")
    lines.extend(
        [
            "",
            "## Method means by motion bin",
            "",
            "| motion bin | method | mean tracking error (deg) | improvement vs raw (deg) | improvement vs persistence (deg) | rows |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["method_means_by_motion_bin"]:
        lines.append(
            "| {motion_bin} | {method} | {mean_tracking_error_deg:.4g} | {mean_improvement_vs_raw_deg:.4g} | {mean_improvement_vs_persistence_deg:.4g} | {row_count} |".format(
                **row
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_motion_stratified_private_accad_eval(
    config: dict[str, Any],
    *,
    source_data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    prepare: bool = True,
) -> dict[str, Any]:
    root = _as_path(
        source_data_root
        or config.get("source_data_root")
        or config.get("data_root", "D:/Uni-Data/ACCAD")
    )
    out_dir = _as_path(
        output_dir or config.get("output_dir", "runs/motion_stratified_private_accad_eval")
    )
    segments_dir = _as_path(config.get("segments_dir", out_dir / "segments"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(config.get("dataset_name", "ACCAD"))

    if prepare:
        window_report = prepare_windows(
            data_root=root,
            output_dir=segments_dir,
            report_path=out_dir / "window_selection_report.json",
            manifest_path=out_dir / "window_manifest.csv",
            dataset_subset=str(config.get("dataset_subset", "")),
            frame_rate=int(config.get("frame_rate", 20)),
            num_joints=int(config.get("num_joints", 23)),
            segment_frames=int(config.get("segment_frames", 80)),
            stride_frames=int(config.get("stride_frames", 40)),
            max_files=(
                None
                if config.get("max_files") is None
                else int(config["max_files"])
            ),
            max_segments=int(config.get("max_segments", 48)),
            selection=str(config.get("selection", "top-motion")),
            min_motion_deg_per_frame=float(
                config.get("min_motion_deg_per_frame", 0.0)
            ),
            max_per_file=(
                None
                if config.get("max_per_file") is None
                else int(config["max_per_file"])
            ),
            clean=not bool(config.get("keep_existing_segments", False)),
        )
    else:
        window_report = json.loads(
            (out_dir / "window_selection_report.json").read_text(encoding="utf-8")
        )

    grouped_windows = _group_windows_by_motion_bin(window_report)
    bin_data_dirs = {
        bin_name: _copy_motion_bin_segments(windows, bin_name, out_dir)
        for bin_name, windows in grouped_windows.items()
    }
    segment_diagnostics_path = out_dir / "motion_bin_segment_diagnostics.json"
    segment_diagnostics = _validate_motion_bin_segments(
        grouped_windows,
        bin_data_dirs,
        segment_diagnostics_path,
        frame_rate=int(config.get("frame_rate", 20)),
        num_joints=int(config.get("num_joints", 23)),
        min_frames=int(config.get("min_frames", 2)),
    )
    methods = tuple(
        str(method)
        for method in config.get("benchmark_methods", list(DEFAULT_METHODS))
    )
    seeds = _int_list(config.get("benchmark_seeds"), [int(config.get("seed", 19))])
    particle_counts = _int_list(
        config.get("benchmark_num_particles"),
        [int(config.get("num_particles", 128))],
    )
    noise_grid = _float_list(
        config.get("benchmark_noise_deg"),
        [float(config.get("noise_deg", 10.0))],
    )
    occlusion_grid = _float_list(
        config.get("benchmark_occlusion_prob"),
        [float(config.get("occlusion_prob", 0.25))],
    )

    runs = []
    all_metric_rows: list[dict[str, Any]] = []
    all_transition_rows: list[dict[str, Any]] = []
    for bin_name, bin_data_dir in bin_data_dirs.items():
        for seed in seeds:
            for num_particles in particle_counts:
                run_name = f"{bin_name}_seed_{seed}_particles_{num_particles}"
                run_dir = out_dir / "benchmarks" / run_name
                run_config = _benchmark_config(
                    config,
                    bin_data_dir=bin_data_dir,
                    output_dir=run_dir,
                    seed=seed,
                    num_particles=num_particles,
                )
                benchmark_summary = run_first_results_benchmark(
                    run_config,
                    run_dir,
                    methods=methods,
                    noise_grid=noise_grid,
                    occlusion_grid=occlusion_grid,
                )
                run = {
                    "motion_bin": bin_name,
                    "seed": seed,
                    "num_particles": num_particles,
                    "output_dir": str(run_dir),
                    "best_method": benchmark_summary["best_method"],
                    "best_tracking_error_deg": benchmark_summary[
                        "best_tracking_error_deg"
                    ],
                }
                runs.append(run)
                for row in _read_csv(run_dir / "benchmark_metrics.csv"):
                    all_metric_rows.append(_copy_row_with_run(row, run, bin_name))
                for row in _read_csv(run_dir / "transition_metrics.csv"):
                    all_transition_rows.append(_copy_row_with_run(row, run, bin_name))

    aggregate_metrics_path = out_dir / "aggregate_benchmark_metrics_by_motion_bin.csv"
    aggregate_transition_path = out_dir / "aggregate_transition_metrics_by_motion_bin.csv"
    aggregate_transition_means_path = out_dir / "aggregate_transition_means_by_motion_bin.csv"
    method_means_by_motion_path = out_dir / "aggregate_method_means_by_motion_bin.csv"
    method_means_by_grid_motion_path = out_dir / "aggregate_method_means_by_noise_occlusion_motion.csv"
    robustness_summary_path = out_dir / "robustness_summary_by_motion_bin.csv"
    particle_collapse_summary_path = out_dir / "particle_collapse_summary_by_motion_bin.csv"
    transition_tracking_diagnostics_path = out_dir / "transition_tracking_diagnostics_by_motion_bin.csv"
    method_means_by_motion = _aggregate_method_means(
        all_metric_rows,
        ("motion_bin", "method"),
    )
    method_means_by_grid_motion = _aggregate_method_means(
        all_metric_rows,
        ("motion_bin", "noise_deg", "occlusion_prob", "method"),
    )
    transition_means_by_motion = _aggregate_transition_means(all_transition_rows)
    robustness_summary = _robustness_summary(method_means_by_grid_motion)
    particle_collapse_summary = _particle_collapse_summary(all_metric_rows)
    transition_tracking_diagnostics = _transition_tracking_diagnostics(
        method_means_by_motion,
        transition_means_by_motion,
    )
    write_csv(aggregate_metrics_path, all_metric_rows)
    write_csv(aggregate_transition_path, all_transition_rows)
    write_csv(aggregate_transition_means_path, transition_means_by_motion)
    write_csv(method_means_by_motion_path, method_means_by_motion)
    write_csv(method_means_by_grid_motion_path, method_means_by_grid_motion)
    write_csv(robustness_summary_path, robustness_summary)
    write_csv(particle_collapse_summary_path, particle_collapse_summary)
    write_csv(transition_tracking_diagnostics_path, transition_tracking_diagnostics)

    motion_bin_counts = {
        bin_name: len(windows) for bin_name, windows in grouped_windows.items()
    }
    summary = {
        "dataset_name": dataset_name,
        "source_data_root": str(root),
        "output_dir": str(out_dir),
        "segments_dir": str(segments_dir),
        "methods": list(methods),
        "seeds": seeds,
        "num_particles": particle_counts,
        "noise_deg": noise_grid,
        "occlusion_prob": occlusion_grid,
        "motion_bins": [
            {"name": name, "min_deg_per_frame": lower, "max_deg_per_frame": upper}
            for name, lower, upper in MOTION_BINS
        ],
        "motion_bin_counts": motion_bin_counts,
        "motion_bin_segment_diagnostics": segment_diagnostics,
        "window_report": window_report,
        "runs": runs,
        "method_means_by_motion_bin": method_means_by_motion,
        "method_means_by_noise_occlusion_motion": method_means_by_grid_motion,
        "transition_means_by_motion_bin": transition_means_by_motion,
        "robustness_summary_by_motion_bin": robustness_summary,
        "particle_collapse_summary_by_motion_bin": particle_collapse_summary,
        "transition_tracking_diagnostics_by_motion_bin": transition_tracking_diagnostics,
        "outputs": {
            "aggregate_benchmark_metrics_by_motion_bin": str(aggregate_metrics_path),
            "aggregate_transition_metrics_by_motion_bin": str(aggregate_transition_path),
            "aggregate_transition_means_by_motion_bin": str(aggregate_transition_means_path),
            "aggregate_method_means_by_motion_bin": str(method_means_by_motion_path),
            "aggregate_method_means_by_noise_occlusion_motion": str(method_means_by_grid_motion_path),
            "robustness_summary_by_motion_bin": str(robustness_summary_path),
            "particle_collapse_summary_by_motion_bin": str(particle_collapse_summary_path),
            "transition_tracking_diagnostics_by_motion_bin": str(transition_tracking_diagnostics_path),
            "motion_bin_segment_diagnostics": str(segment_diagnostics_path),
            "markdown_summary": str(out_dir / "motion_stratified_private_accad_eval_summary.md"),
        },
    }
    write_json(out_dir / "motion_stratified_private_accad_eval_summary.json", summary)
    _write_markdown_summary(
        out_dir / "motion_stratified_private_accad_eval_summary.md",
        summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run motion-stratified private AMASS evaluation into ignored local output folders."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--data-root", default=None, help="Override source AMASS root from config.")
    parser.add_argument("--output", default=None, help="Override output directory from config.")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse an existing window_selection_report.json and segments_dir.",
    )
    args = parser.parse_args()

    config = load_motion_stratified_config(args.config)
    summary = run_motion_stratified_private_accad_eval(
        config,
        source_data_root=args.data_root,
        output_dir=args.output,
        prepare=not args.skip_prepare,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
