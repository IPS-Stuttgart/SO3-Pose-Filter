from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_filter.data import load_dataset, split_sequences  # noqa: E402
from pose_filter.evaluation import (  # noqa: E402
    robustness_rows,
    transition_metric_rows,
    write_csv,
    write_json,
)
from pose_filter.experiment import load_config  # noqa: E402
from pose_filter.plotting import heatmap_svg, line_plot_svg  # noqa: E402
from pose_filter.transitions import build_transition_model  # noqa: E402

METHODS = ("raw", "persistence", "gaussian_rw", "pyrecest_pf")


def _float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(value)]


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _grid_key(row: dict[str, Any]) -> tuple[float, float]:
    return (float(row["noise_deg"]), float(row["occlusion_prob"]))


def _method_row(
    method: str,
    source: dict[str, Any],
    raw_source: dict[str, Any],
    tracking_metric: str,
    transition_model: str,
    filter_backend: str,
    num_particles: int,
) -> dict[str, Any]:
    tracking_error = float(source[tracking_metric])
    raw_error = float(raw_source["observed_error_deg"])
    persistence_error = float(raw_source["persistence_error_deg"])
    return {
        "method": method,
        "noise_deg": float(source["noise_deg"]),
        "occlusion_prob": float(source["occlusion_prob"]),
        "tracking_error_deg": tracking_error,
        "raw_observed_error_deg": raw_error,
        "persistence_error_deg": persistence_error,
        "improvement_vs_raw_deg": raw_error - tracking_error,
        "improvement_vs_persistence_deg": persistence_error - tracking_error,
        "mean_ess": float(source.get("mean_ess", float("nan"))),
        "transition_model": transition_model,
        "filter_backend": filter_backend,
        "source_metric": tracking_metric,
        "num_particles": int(num_particles),
    }


def _closest(values: list[float], target: float) -> float:
    return min(values, key=lambda value: abs(value - target))


def _method_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        method: _mean(
            [
                float(row["tracking_error_deg"])
                for row in rows
                if row["method"] == method
            ]
        )
        for method in sorted({row["method"] for row in rows})
    }


def _representative_rows(
    rows: list[dict[str, Any]],
    noise_deg: float,
    occlusion_prob: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    noise = _closest(sorted({float(row["noise_deg"]) for row in rows}), noise_deg)
    occlusion = _closest(
        sorted({float(row["occlusion_prob"]) for row in rows}), occlusion_prob
    )
    return [
        row
        for row in rows
        if float(row["noise_deg"]) == noise
        and float(row["occlusion_prob"]) == occlusion
    ]


def _acceptance(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    representative = _representative_rows(
        rows,
        float(config["noise_deg"]),
        float(config["occlusion_prob"]),
    )
    representative_by_method = {row["method"]: row for row in representative}
    pyrecest = representative_by_method.get("pyrecest_pf")
    return {
        "representative_noise_deg": representative[0]["noise_deg"]
        if representative
        else None,
        "representative_occlusion_prob": representative[0]["occlusion_prob"]
        if representative
        else None,
        "pyrecest_pf_beats_raw_at_representative": (
            bool(pyrecest["improvement_vs_raw_deg"] > 0.0)
            if pyrecest is not None
            else None
        ),
        "pyrecest_pf_beats_persistence_at_representative": (
            bool(pyrecest["improvement_vs_persistence_deg"] > 0.0)
            if pyrecest is not None
            else None
        ),
        "pyrecest_pf_beats_raw_any": any(
            row["method"] == "pyrecest_pf"
            and float(row["improvement_vs_raw_deg"]) > 0.0
            for row in rows
        ),
        "pyrecest_pf_beats_persistence_any": any(
            row["method"] == "pyrecest_pf"
            and float(row["improvement_vs_persistence_deg"]) > 0.0
            for row in rows
        ),
    }


def _write_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    methods: tuple[str, ...],
    config: dict[str, Any],
) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    plot_method = str(
        config.get(
            "benchmark_heatmap_method",
            "pyrecest_pf" if "pyrecest_pf" in methods else "gaussian_rw",
        )
    )
    heatmap_rows = [row for row in rows if row["method"] == plot_method]
    heatmap_path = plots_dir / "tracking_error_heatmap.svg"
    heatmap_svg(
        heatmap_path,
        heatmap_rows,
        x_key="noise_deg",
        y_key="occlusion_prob",
        value_key="tracking_error_deg",
        title=f"{plot_method} tracking error",
        x_label="measurement noise (deg)",
        y_label="occlusion probability",
        value_label="deg",
    )

    occlusion_values = sorted({float(row["occlusion_prob"]) for row in rows})
    curve_occlusion = _closest(
        occlusion_values,
        float(config.get("benchmark_plot_occlusion_prob", config["occlusion_prob"])),
    )
    curve_rows = [
        row for row in rows if float(row["occlusion_prob"]) == curve_occlusion
    ]
    series: dict[str, list[tuple[float, float]]] = {}
    for method in methods:
        points = [
            (float(row["noise_deg"]), float(row["tracking_error_deg"]))
            for row in curve_rows
            if row["method"] == method
        ]
        if points:
            series[method] = sorted(points, key=lambda item: item[0])
    curve_path = plots_dir / "filter_vs_baselines.svg"
    line_plot_svg(
        curve_path,
        series,
        title=f"Filter vs baselines at occlusion={curve_occlusion:g}",
        x_label="measurement noise (deg)",
        y_label="tracking error (deg)",
    )
    return {
        "tracking_error_heatmap": str(heatmap_path),
        "filter_vs_baselines": str(curve_path),
    }


def run_first_results_benchmark(
    config: dict[str, Any],
    output_dir: str | Path,
    methods: tuple[str, ...] | None = None,
    noise_grid: list[float] | None = None,
    occlusion_grid: list[float] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 0))
    num_particles = int(config["num_particles"])
    noise_grid = noise_grid or _float_list(
        config.get("benchmark_noise_deg"),
        _float_list(config.get("robustness_noise_deg"), [float(config["noise_deg"])]),
    )
    occlusion_grid = occlusion_grid or _float_list(
        config.get("benchmark_occlusion_prob"),
        _float_list(
            config.get("robustness_occlusion_prob"), [float(config["occlusion_prob"])]
        ),
    )
    config_methods = config.get("benchmark_methods") if methods is None else None
    if config_methods is None and methods is None:
        methods = METHODS
    elif config_methods is not None:
        methods = tuple(str(method) for method in config_methods)
    if methods is None:
        raise RuntimeError("benchmark methods were not initialized")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown benchmark methods: {', '.join(unknown)}")

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
        "gaussian_rw",
        train,
        process_noise_deg=config.get("process_noise_deg"),
    )
    proposal_gain = float(config.get("proposal_gain", 0.2))
    confidence_noise_std = float(config.get("confidence_noise_std", 0.0))
    min_confidence = float(config.get("min_confidence", 0.2))
    factorized_update = bool(config.get("factorized_update", True))
    resample_threshold = float(config.get("resample_threshold", 0.5))
    base_rows = robustness_rows(
        test,
        model,
        noise_grid,
        occlusion_grid,
        num_particles,
        seed,
        proposal_gain=proposal_gain,
        confidence_noise_std=confidence_noise_std,
        min_confidence=min_confidence,
        factorized_update=factorized_update,
        resample_threshold=resample_threshold,
        filter_backend="numpy",
    )
    pyrecest_rows_by_key: dict[tuple[float, float], dict[str, Any]] = {}
    if "pyrecest_pf" in methods:
        pyrecest_rows = robustness_rows(
            test,
            model,
            noise_grid,
            occlusion_grid,
            num_particles,
            seed,
            proposal_gain=proposal_gain,
            confidence_noise_std=confidence_noise_std,
            min_confidence=min_confidence,
            factorized_update=factorized_update,
            resample_threshold=resample_threshold,
            filter_backend="pyrecest",
        )
        pyrecest_rows_by_key = {_grid_key(row): row for row in pyrecest_rows}

    rows: list[dict[str, Any]] = []
    for base_row in base_rows:
        key = _grid_key(base_row)
        if "raw" in methods:
            rows.append(
                _method_row(
                    "raw",
                    base_row,
                    base_row,
                    "observed_error_deg",
                    "none",
                    "none",
                    num_particles,
                )
            )
        if "persistence" in methods:
            rows.append(
                _method_row(
                    "persistence",
                    base_row,
                    base_row,
                    "persistence_error_deg",
                    "deterministic_persistence",
                    "none",
                    num_particles,
                )
            )
        if "gaussian_rw" in methods:
            rows.append(
                _method_row(
                    "gaussian_rw",
                    base_row,
                    base_row,
                    "filter_error_deg",
                    "gaussian_rw",
                    "numpy",
                    num_particles,
                )
            )
        if "pyrecest_pf" in methods:
            rows.append(
                _method_row(
                    "pyrecest_pf",
                    pyrecest_rows_by_key[key],
                    base_row,
                    "filter_error_deg",
                    "gaussian_rw",
                    "pyrecest",
                    num_particles,
                )
            )

    metrics_path = output_dir / "benchmark_metrics.csv"
    _write_csv(metrics_path, rows)
    transition_rows = transition_metric_rows(
        "gaussian_rw",
        model,
        test,
        rollout_horizon=int(config.get("rollout_horizon", 10)),
    )
    write_csv(output_dir / "transition_metrics.csv", transition_rows)
    plots = _write_plots(output_dir, rows, methods, config)

    means_by_method = _method_means(rows)
    best_method = min(
        means_by_method,
        key=lambda method: means_by_method[method],
    )
    summary = {
        "methods": list(methods),
        "noise_deg": noise_grid,
        "occlusion_prob": occlusion_grid,
        "num_sequences": len(sequences),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "num_particles": num_particles,
        "row_count": len(rows),
        "means_by_method": means_by_method,
        "best_method": best_method,
        "best_tracking_error_deg": means_by_method[best_method],
        "acceptance": _acceptance(rows, config),
        "transition_metrics": transition_rows,
        "outputs": {
            "benchmark_metrics": str(metrics_path),
            "transition_metrics": str(output_dir / "transition_metrics.csv"),
            "plots": plots,
        },
    }
    write_json(output_dir / "first_results_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the compact first-results benchmark across noise/occlusion."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to config.benchmark_output_dir.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=None,
        help="Benchmark methods to report.",
    )
    parser.add_argument(
        "--noise-deg",
        nargs="+",
        type=float,
        default=None,
        help="Override benchmark noise grid.",
    )
    parser.add_argument(
        "--occlusion-prob",
        nargs="+",
        type=float,
        default=None,
        help="Override benchmark occlusion grid.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(
        args.output
        or config.get("benchmark_output_dir", "runs/first_results_benchmark")
    )
    methods = tuple(args.methods) if args.methods is not None else None
    payload = run_first_results_benchmark(
        config,
        output_dir,
        methods=methods,
        noise_grid=args.noise_deg,
        occlusion_grid=args.occlusion_prob,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
