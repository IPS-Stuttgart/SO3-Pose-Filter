from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "build_accuracy_leaderboard",
        root / "scripts" / "build_accuracy_leaderboard.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load scripts/build_accuracy_leaderboard.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AccuracyLeaderboardTest(unittest.TestCase):
    def test_motion_leaderboard_writes_condition_and_summary_outputs(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact"
            artifact.mkdir()
            path = artifact / "aggregate_method_means_by_noise_occlusion_motion.csv"
            fieldnames = [
                "motion_bin",
                "noise_deg",
                "occlusion_prob",
                "method",
                "mean_tracking_error_deg",
                "std_tracking_error_deg",
                "sem_tracking_error_deg",
                "mean_improvement_vs_raw_deg",
                "std_improvement_vs_raw_deg",
                "sem_improvement_vs_raw_deg",
                "mean_improvement_vs_persistence_deg",
                "std_improvement_vs_persistence_deg",
                "sem_improvement_vs_persistence_deg",
                "row_count",
            ]
            rows = [
                ("low_motion", "10.0", "0.0", "raw", "10.0", "0.0", "2.0"),
                ("low_motion", "10.0", "0.0", "persistence", "12.0", "-2.0", "0.0"),
                ("low_motion", "10.0", "0.0", "gaussian_rw", "8.0", "2.0", "4.0"),
                ("low_motion", "20.0", "0.5", "raw", "20.0", "0.0", "-5.0"),
                ("low_motion", "20.0", "0.5", "persistence", "15.0", "5.0", "0.0"),
                ("low_motion", "20.0", "0.5", "gaussian_rw", "12.0", "8.0", "3.0"),
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for motion_bin, noise, occlusion, method, tracking, raw_delta, persistence_delta in rows:
                    writer.writerow(
                        {
                            "motion_bin": motion_bin,
                            "noise_deg": noise,
                            "occlusion_prob": occlusion,
                            "method": method,
                            "mean_tracking_error_deg": tracking,
                            "std_tracking_error_deg": "0.0",
                            "sem_tracking_error_deg": "0.0",
                            "mean_improvement_vs_raw_deg": raw_delta,
                            "std_improvement_vs_raw_deg": "0.0",
                            "sem_improvement_vs_raw_deg": "0.0",
                            "mean_improvement_vs_persistence_deg": persistence_delta,
                            "std_improvement_vs_persistence_deg": "0.0",
                            "sem_improvement_vs_persistence_deg": "0.0",
                            "row_count": "3",
                        }
                    )

            leaderboard_rows = module.build_leaderboard(
                detector_runs=[],
                motion_runs=[module.MotionRunSpec(dataset="KIT", path=artifact)],
                detector_dataset="detector_hmr",
            )
            self.assertIn("noise_deg", module.LEADERBOARD_COLUMNS)
            best_by_condition = {(row["noise_deg"], row["occlusion_prob"]): row["method"] for row in leaderboard_rows if row["rank"] == 1}
            self.assertEqual(best_by_condition, {("10.0", "0.0"): "gaussian_rw", ("20.0", "0.5"): "gaussian_rw"})

            output = root / "out"
            outputs = module.write_outputs(output, leaderboard_rows)
            for key in [
                "csv",
                "json",
                "markdown",
                "latex",
                "paper_summary_csv",
                "paper_summary_json",
                "paper_summary_markdown",
                "paper_summary_latex",
                "sanity_report_json",
                "sanity_report_markdown",
            ]:
                self.assertTrue(Path(outputs[key]).is_file(), key)

            summary = json.loads(Path(outputs["paper_summary_json"]).read_text(encoding="utf-8"))
            gaussian = next(row for row in summary["rows"] if row["method"] == "gaussian_rw")
            self.assertEqual(gaussian["condition_count"], 2)
            self.assertEqual(gaussian["win_count"], 2)
            self.assertAlmostEqual(gaussian["mean_tracking_error_deg"], 10.0)

            sanity = json.loads(Path(outputs["sanity_report_json"]).read_text(encoding="utf-8"))
            self.assertEqual(sanity["missing_baseline_conditions"], [])
            self.assertEqual(sanity["duplicate_context_rows"], [])


if __name__ == "__main__":
    unittest.main()
