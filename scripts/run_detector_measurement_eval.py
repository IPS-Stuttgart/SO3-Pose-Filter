from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_filter.detector_evaluation import load_detector_eval_config, run_detector_measurement_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run filtering with real detector/SMPL-fitting measurements.")
    parser.add_argument("--config", required=True, help="Path to detector-measurement evaluation JSON config.")
    args = parser.parse_args()
    summary = run_detector_measurement_eval(load_detector_eval_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
