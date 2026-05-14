from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_filter.experiment import load_config, run_experiment  # noqa: E402


def run_realistic_experiment(config: dict) -> dict:
    """Run the standard experiment with config-defined realistic measurements."""

    return run_experiment(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an SO(3)^K benchmark with config-defined realistic synthetic measurements."
    )
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    args = parser.parse_args()

    summary = run_realistic_experiment(load_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
