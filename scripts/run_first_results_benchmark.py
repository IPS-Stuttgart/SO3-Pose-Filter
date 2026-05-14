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

from pose_filter.constant_velocity import ConstantVelocityTransition  # noqa: E402
from pose_filter.data import load_dataset, split_sequences  # noqa: E402
from pose_filter.evaluation import (  # noqa: E402
    robustness_rows,
    transition_metric_rows,
    write_csv,
    write_json,
)
from pose_filter.experiment import load_config  # noqa: E402
from pose_filter.measurement_config import MeasurementRealismConfig  # noqa: E402
from pose_filter.plotting import heatmap_svg, line_plot_svg  # noqa: E402
from pose_filter.smoothing import SmootherConfig  # noqa: E402
from pose_filter.transitions import build_transition_model  # noqa: E402

METHODS = (
    "raw",
    "persistence",
    "smoother_ema",
    "smoother_chordal",
    "savgol_tangent",
    "deterministic_persistence_pf",
    "noisy_persistence_pf",
    "constant_velocity",
    "gaussian_rw",
    "adaptive_gaussian_rw",
    "noise_adaptive_selector",
    "pyrecest_pf",
    "mlp_delta",
    "pyrecest_mlp_pf",
    "history_mlp_delta",
    "gru_delta",
)
SMOOTHER_METHODS = {
    "smoother_ema": "smoother_ema_error_deg",
    "smoother_chordal": "smoother_chordal_error_deg",
    "savgol_tangent": "savgol_tangent_error_deg",
}


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
    filter_diagnostics = {
        "mean_ess": float(source.get("mean_ess", float("nan"))),
        "min_ess": float(source.get("min_ess", float("nan"))),
        "final_ess": float(source.get("final_ess", float("nan"))),
        "resample_count": float(source.get("resample_count", float("nan"))),
        "resample_fraction": float(source.get("resample_fraction", float("nan"))),
        "mean_particle_spread_deg": float(source.get("mean_particle_spread_deg", float("nan"))),
        "min_particle_spread_deg": float(source.get("min_particle_spread_deg", float("nan"))),
        "final_particle_spread_deg": float(source.get("final_particle_spread_deg", float("nan"))),
        "collapse_fraction": float(source.get("collapse_fraction", float("nan"))),
        "filter_reappeared_joint_error_deg": float(source.get("filter_reappeared_joint_error_deg", float("nan"))),
        "persistence_reappeared_joint_error_deg": float(source.get("persistence_reappeared_joint_error_deg", float("nan"))),
        "reappeared_joint_count": float(source.get("reappeared_joint_count", float("nan"))),
    }
    if filter_backend not in {"numpy", "pyrecest"}:
        filter_diagnostics = {key: float("nan") for key in filter_diagnostics}
    return {
        "method": method,
        "noise_deg": float(source["noise_deg"]),
        "occlusion_prob": float(source["occlusion_prob"]),
        "occlusion_model": source.get("occlusion_model", "iid"),
        "outlier_prob": float(source.get("outlier_prob", 0.0)),
        "outlier_fraction": float(source.get("outlier_fraction", float("nan"))),
        "confidence_calibrated_noise": bool(source.get("confidence_calibrated_noise", False)),
        "mean_joint_noise_sigma_deg": float(source.get("mean_joint_noise_sigma_deg", float("nan"))),
        "tracking_error_deg": tracking_error,
        "raw_observed_error_deg": raw_error,
        "persistence_error_deg": persistence_error,
        "improvement_vs_raw_deg": raw_error - tracking_error,
        "improvement_vs_persistence_deg": persistence_error - tracking_error,
        "transition_model": transition_model,
        "filter_backend": filter_backend,
        "source_metric": tracking_metric,
        "num_particles": int(num_particles),
        **filter_diagnostics,
    }


def _noise_adaptive_selector_row(base_row: dict[str, Any], *, threshold_deg: float, num_particles: int) -> dict[str, Any]:
    use_gaussian = float(base_row["noise_deg"]) <= float(threshold_deg)
    source_metric = "filter_error_deg" if use_gaussian else "persistence_error_deg"
    row = _method_row(
        "noise_adaptive_selector",
        base_row,
        base_row,
        source_metric,
        "noise_adaptive_selector",
        "selector",
        num_particles,
    )
    selected = "gaussian_rw" if use_gaussian else "persistence"
    row["source_metric"] = f"{selected}:{source_metric}"
    return row


def _closest(values: list[float], target: float) -> float:
    return min(values, key=lambda value: abs(value - target))


def _method_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        method: _mean([float(row["tracking_error_deg"]) for row in rows if row["method"] == method])
        for method in sorted({row["method"] for row in rows})
    }


def _representative_rows(rows: list[dict[str, Any]], noise_deg: float, occlusion_prob: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    noise = _closest(sorted({float(row["noise_deg"]) for row in rows}), noise_deg)
    occlusion = _closest(sorted({float(row["occlusion_prob"]) for row in rows}), occlusion_prob)
    return [row for row in rows if float(row["noise_deg"]) == noise and float(row["occlusion_prob"]) == occlusion]


def _acceptance(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    representative = _representative_rows(rows, float(config["noise_deg"]), float(config["occlusion_prob"]))
    representative_by_method = {row["method"]: row for row in representative}
    pyrecest = representative_by_method.get("pyrecest_pf")
    target_method = str(
        config.get(
            "benchmark_acceptance_method",
            "gru_delta"
            if any(row["method"] == "gru_delta" for row in rows)
            else "history_mlp_delta"
            if any(row["method"] == "history_mlp_delta" for row in rows)
            else "pyrecest_mlp_pf"
            if any(row["method"] == "pyrecest_mlp_pf" for row in rows)
            else "pyrecest_pf",
        )
    )
    target = representative_by_method.get(target_method)
    return {
        "target_method": target_method,
        "representative_noise_deg": representative[0]["noise_deg"] if representative else None,
        "representative_occlusion_prob": representative[0]["occlusion_prob"] if representative else None,
        "target_beats_raw_at_representative": bool(target["improvement_vs_raw_deg"] > 0.0) if target is not None else None,
        "target_beats_persistence_at_representative": bool(target["improvement_vs_persistence_deg"] > 0.0) if target is not None else None,
        "target_beats_raw_any": any(row["method"] == target_method and float(row["improvement_vs_raw_deg"]) > 0.0 for row in rows),
        "target_beats_persistence_any": any(row["method"] == target_method and float(row["improvement_vs_persistence_deg"]) > 0.0 for row in rows),
        "pyrecest_pf_beats_raw_at_representative": bool(pyrecest["improvement_vs_raw_deg"] > 0.0) if pyrecest is not None else None,
        "pyrecest_pf_beats_persistence_at_representative": bool(pyrecest["improvement_vs_persistence_deg"] > 0.0) if pyrecest is not None else None,
        "pyrecest_pf_beats_raw_any": any(row["method"] == "pyrecest_pf" and float(row["improvement_vs_raw_deg"]) > 0.0 for row in rows),
        "pyrecest_pf_beats_persistence_any": any(row["method"] == "pyrecest_pf" and float(row["improvement_vs_persistence_deg"]) > 0.0 for row in rows),
    }


def _write_plots(output_dir: Path, rows: list[dict[str, Any]], methods: tuple[str, ...], config: dict[str, Any]) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    plot_method = str(
        config.get(
            "benchmark_heatmap_method",
            "noise_adaptive_selector"
            if "noise_adaptive_selector" in methods
            else "adaptive_gaussian_rw"
            if "adaptive_gaussian_rw" in methods
            else "pyrecest_mlp_pf"
            if "pyrecest_mlp_pf" in methods
            else "gru_delta"
            if "gru_delta" in methods
            else "history_mlp_delta"
            if "history_mlp_delta" in methods
            else "constant_velocity"
            if "constant_velocity" in methods
            else "pyrecest_pf"
            if "pyrecest_pf" in methods
            else "gaussian_rw",
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
    curve_occlusion = _closest(occlusion_values, float(config.get("benchmark_plot_occlusion_prob", config["occlusion_prob"])))
    curve_rows = [row for row in rows if float(row["occlusion_prob"]) == curve_occlusion]
    series: dict[str, list[tuple[float, float]]] = {}
    for method in methods:
        points = [(float(row["noise_deg"]), float(row["tracking_error_deg"])) for row in curve_rows if row["method"] == method]
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
    return {"tracking_error_heatmap": str(heatmap_path), "filter_vs_baselines": str(curve_path)}


def _process_noise_cap(config: dict[str, Any]) -> float | None:
    process_noise_deg = config.get("process_noise_deg")
    if process_noise_deg is None:
        return None
    return np.radians(float(process_noise_deg))


def _smoother_config(config: dict[str, Any]) -> SmootherConfig:
    return SmootherConfig(
        ema_alpha=float(config.get("smoother_ema_alpha", 0.35)),
        chordal_window=int(config.get("smoother_chordal_window", 5)),
        tangent_savgol_window=int(config.get("savgol_tangent_window", 7)),
        tangent_savgol_degree=int(config.get("savgol_tangent_degree", 2)),
    )


def _run_robustness(
    test,
    model,
    noise_grid: list[float],
    occlusion_grid: list[float],
    num_particles: int,
    seed: int,
    *,
    proposal_gain: float,
    confidence_noise_std: float,
    min_confidence: float,
    factorized_update: bool,
    resample_threshold: float,
    filter_backend: str,
    smoother_config: SmootherConfig,
    measurement_config: MeasurementRealismConfig,
) -> list[dict[str, Any]]:
    return robustness_rows(
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
        filter_backend=filter_backend,
        smoother_config=smoother_config,
        measurement_config=measurement_config,
    )


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
    noise_grid = noise_grid or _float_list(config.get("benchmark_noise_deg"), _float_list(config.get("robustness_noise_deg"), [float(config["noise_deg"])]))
    occlusion_grid = occlusion_grid or _float_list(config.get("benchmark_occlusion_prob"), _float_list(config.get("robustness_occlusion_prob"), [float(config["occlusion_prob"])]))
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

    proposal_gain = float(config.get("proposal_gain", 0.2))
    confidence_noise_std = float(config.get("confidence_noise_std", 0.0))
    min_confidence = float(config.get("min_confidence", 0.2))
    factorized_update = bool(config.get("factorized_update", True))
    resample_threshold = float(config.get("resample_threshold", 0.5))
    smoother_config = _smoother_config(config)
    measurement_config = MeasurementRealismConfig.from_mapping(config)

    gaussian_model = build_transition_model("gaussian_rw", train, process_noise_deg=config.get("process_noise_deg"), config=config)
    base_rows = _run_robustness(
        test,
        gaussian_model,
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
        smoother_config=smoother_config,
        measurement_config=measurement_config,
    )

    noise_adaptive_threshold_deg = float(config.get("noise_adaptive_selector_threshold_deg", 10.0))
    collapse_ablation_proposal_gain = float(config.get("collapse_ablation_proposal_gain", 0.0))
    model_rows_by_method: dict[str, dict[tuple[float, float], dict[str, Any]]] = {}
    built_models: dict[str, Any] = {"gaussian_rw": gaussian_model}

    def add_model_rows(method: str, model_name: str, backend: str = "numpy", gain: float | None = None) -> None:
        if method not in methods:
            return
        if model_name == "constant_velocity":
            model = ConstantVelocityTransition.fit(train, max_std_rad=_process_noise_cap(config))
        else:
            model = build_transition_model(model_name, train, process_noise_deg=config.get("process_noise_deg"), config=config)
        built_models[model_name] = model
        rows = _run_robustness(
            test,
            model,
            noise_grid,
            occlusion_grid,
            num_particles,
            seed,
            proposal_gain=proposal_gain if gain is None else gain,
            confidence_noise_std=confidence_noise_std,
            min_confidence=min_confidence,
            factorized_update=factorized_update,
            resample_threshold=resample_threshold,
            filter_backend=backend,
            smoother_config=smoother_config,
            measurement_config=measurement_config,
        )
        model_rows_by_method[method] = {_grid_key(row): row for row in rows}

    add_model_rows("deterministic_persistence_pf", "deterministic_persistence", gain=collapse_ablation_proposal_gain)
    add_model_rows("noisy_persistence_pf", "noisy_persistence", gain=collapse_ablation_proposal_gain)
    add_model_rows("constant_velocity", "constant_velocity")
    add_model_rows("adaptive_gaussian_rw", "adaptive_gaussian_rw")
    if "pyrecest_pf" in methods:
        rows = _run_robustness(
            test,
            gaussian_model,
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
            smoother_config=smoother_config,
            measurement_config=measurement_config,
        )
        model_rows_by_method["pyrecest_pf"] = {_grid_key(row): row for row in rows}
    if "mlp_delta" in methods or "pyrecest_mlp_pf" in methods:
        mlp_model = build_transition_model("mlp_delta", train, process_noise_deg=config.get("process_noise_deg"), config=config)
        built_models["mlp_delta"] = mlp_model
        if "mlp_delta" in methods:
            rows = _run_robustness(test, mlp_model, noise_grid, occlusion_grid, num_particles, seed, proposal_gain=proposal_gain, confidence_noise_std=confidence_noise_std, min_confidence=min_confidence, factorized_update=factorized_update, resample_threshold=resample_threshold, filter_backend="numpy", smoother_config=smoother_config, measurement_config=measurement_config)
            model_rows_by_method["mlp_delta"] = {_grid_key(row): row for row in rows}
        if "pyrecest_mlp_pf" in methods:
            rows = _run_robustness(test, mlp_model, noise_grid, occlusion_grid, num_particles, seed, proposal_gain=proposal_gain, confidence_noise_std=confidence_noise_std, min_confidence=min_confidence, factorized_update=factorized_update, resample_threshold=resample_threshold, filter_backend="pyrecest", smoother_config=smoother_config, measurement_config=measurement_config)
            model_rows_by_method["pyrecest_mlp_pf"] = {_grid_key(row): row for row in rows}
    add_model_rows("history_mlp_delta", "history_mlp_delta")
    add_model_rows("gru_delta", "gru_delta")

    rows: list[dict[str, Any]] = []
    for base_row in base_rows:
        key = _grid_key(base_row)
        if "raw" in methods:
            rows.append(_method_row("raw", base_row, base_row, "observed_error_deg", "none", "none", num_particles))
        if "persistence" in methods:
            rows.append(_method_row("persistence", base_row, base_row, "persistence_error_deg", "deterministic_persistence", "none", num_particles))
        for smoother_method, metric in SMOOTHER_METHODS.items():
            if smoother_method in methods:
                rows.append(_method_row(smoother_method, base_row, base_row, metric, smoother_method, "offline_smoother", num_particles))
        if "gaussian_rw" in methods:
            rows.append(_method_row("gaussian_rw", base_row, base_row, "filter_error_deg", "gaussian_rw", "numpy", num_particles))
        if "noise_adaptive_selector" in methods:
            rows.append(_noise_adaptive_selector_row(base_row, threshold_deg=noise_adaptive_threshold_deg, num_particles=num_particles))
        for method, transition_model, backend in [
            ("deterministic_persistence_pf", "deterministic_persistence", "numpy"),
            ("noisy_persistence_pf", "noisy_persistence", "numpy"),
            ("constant_velocity", "constant_velocity", "numpy"),
            ("adaptive_gaussian_rw", "adaptive_gaussian_rw", "numpy"),
            ("pyrecest_pf", "gaussian_rw", "pyrecest"),
            ("mlp_delta", "mlp_delta", "numpy"),
            ("pyrecest_mlp_pf", "mlp_delta", "pyrecest"),
            ("history_mlp_delta", "history_mlp_delta", "numpy"),
            ("gru_delta", "gru_delta", "numpy"),
        ]:
            if method in methods:
                rows.append(_method_row(method, model_rows_by_method[method][key], base_row, "filter_error_deg", transition_model, backend, num_particles))

    metrics_path = output_dir / "benchmark_metrics.csv"
    _write_csv(metrics_path, rows)

    transition_rows = transition_metric_rows("gaussian_rw", gaussian_model, test, rollout_horizon=int(config.get("rollout_horizon", 10)))
    for model_name, model in built_models.items():
        if model_name == "gaussian_rw":
            continue
        transition_rows.extend(transition_metric_rows(model_name, model, test, rollout_horizon=int(config.get("rollout_horizon", 10))))
    write_csv(output_dir / "transition_metrics.csv", transition_rows)
    plots = _write_plots(output_dir, rows, methods, config)

    means_by_method = _method_means(rows)
    best_method = min(means_by_method, key=lambda method: means_by_method[method])
    summary = {
        "methods": list(methods),
        "noise_deg": noise_grid,
        "occlusion_prob": occlusion_grid,
        "measurement_model": measurement_config.to_summary(
            noise_deg=float(config["noise_deg"]),
            occlusion_prob=float(config["occlusion_prob"]),
            confidence_noise_std=confidence_noise_std,
            min_confidence=min_confidence,
        ),
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
    parser = argparse.ArgumentParser(description="Run the compact first-results benchmark across noise/occlusion.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults to config.benchmark_output_dir.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None, help="Benchmark methods to report.")
    parser.add_argument("--noise-deg", nargs="+", type=float, default=None, help="Override benchmark noise grid.")
    parser.add_argument("--occlusion-prob", nargs="+", type=float, default=None, help="Override benchmark occlusion grid.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output or config.get("benchmark_output_dir", "runs/first_results_benchmark"))
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
