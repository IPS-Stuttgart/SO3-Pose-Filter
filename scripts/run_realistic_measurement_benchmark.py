from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_filter.experiment import load_config, run_experiment  # noqa: E402
from pose_filter.measurement_config import (  # noqa: E402
    measurement_realism_kwargs,
    measurement_realism_summary,
)
from pose_filter.realistic_evaluation import realistic_measurement_context  # noqa: E402


def run_realistic_experiment(config: dict) -> dict:
    """Run the standard experiment with realistic synthetic measurements enabled."""

    measurement_kwargs = measurement_realism_kwargs(config)
    with realistic_measurement_context(measurement_kwargs):
        summary = run_experiment(config)

    output_dir = Path(config.get("output_dir", "runs/default"))
    measurement_model = {
        "noise_deg": float(config["noise_deg"]),
        "occlusion_prob": float(config["occlusion_prob"]),
        "confidence_noise_std": float(config.get("confidence_noise_std", 0.0)),
        "min_confidence": float(config.get("min_confidence", 0.2)),
        **measurement_realism_summary(config),
    }
    summary["measurement_model"] = measurement_model
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        persisted = json.loads(summary_path.read_text(encoding="utf-8"))
        persisted["measurement_model"] = measurement_model
        summary_path.write_text(json.dumps(persisted, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an SO(3)^K benchmark with Markov occlusion, outliers, and calibrated confidence noise."
    )
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    args = parser.parse_args()

    summary = run_realistic_experiment(load_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
