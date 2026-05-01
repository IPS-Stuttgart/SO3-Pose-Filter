from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_filter.experiment import load_config, run_experiment  # noqa: E402

DEFAULT_MODELS = ("persistence", "gaussian_rw", "learned_delta")
ALL_MODELS = (
    *DEFAULT_MODELS,
    "constant_velocity",
    "mlp_delta",
    "history_mlp_delta",
    "gru_delta",
)


def _metric(summary: dict, name: str) -> float:
    for row in summary["transition_metrics"]:
        if row["metric"] == name:
            return float(row["value"])
    return float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(
    config: dict, output_root: Path, models: tuple[str, ...] = DEFAULT_MODELS
) -> dict:
    rows: list[dict[str, Any]] = []
    summaries = {}
    for model in models:
        model_config = dict(config)
        model_config["transition_model"] = model
        model_config["output_dir"] = str(output_root / model)
        summary = run_experiment(model_config)
        summaries[model] = summary
        means = summary["filter_metrics_mean"]
        rows.append(
            {
                "model": model,
                "one_step_error_deg": _metric(summary, "one_step_error_deg"),
                "rollout_error_deg": _metric(summary, "rollout_error_deg"),
                "observed_error_deg": float(means["observed_error_deg"]),
                "filter_error_deg": float(means["filter_error_deg"]),
                "persistence_error_deg": float(means["persistence_error_deg"]),
                "mean_ess": float(means["mean_ess"]),
            }
        )

    best = min(rows, key=lambda row: float(row["filter_error_deg"]))
    payload = {
        "models": list(models),
        "best_filter_model": best["model"],
        "best_filter_error_deg": best["filter_error_deg"],
        "rows": rows,
        "experiment_dirs": {model: str(output_root / model) for model in models},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "comparison_metrics.csv", rows)
    (output_root / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all transition baselines from one config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to JSON experiment config."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Sweep output directory. Defaults to config.sweep_output_dir or runs/sweep.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(DEFAULT_MODELS),
        help="Transition models to evaluate.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output = Path(args.output or config.get("sweep_output_dir", "runs/sweep"))
    payload = run_sweep(config, output, models=tuple(args.models))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
