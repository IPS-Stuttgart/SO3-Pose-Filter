from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _path import SCRIPTS  # noqa: F401
from package_paper_artifact import package_artifact


class PackagePaperArtifactTests(unittest.TestCase):
    def test_package_artifact_redacts_runtime_and_markdown_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result_root = root / "result"
            result_root.mkdir()
            (result_root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "runtime": {
                            "python_executable": "/home/user/actions-runners/_work/_tool/Python/bin/python"
                        },
                        "source_data_root": "/home/user/private/ACCAD",
                    }
                ),
                encoding="utf-8",
            )
            (result_root / "motion_stratified_private_accad_eval_summary.md").write_text(
                "- source data root: `/home/user/actions-runners/so3-pose-filter/accad-full-data`\n",
                encoding="utf-8",
            )

            output_dir = root / "paper"
            package_artifact(result_root, output_dir)

            manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runtime"]["python_executable"], "<redacted>")
            self.assertEqual(manifest["source_data_root"], "<redacted>")
            summary = (output_dir / "motion_stratified_private_accad_eval_summary.md").read_text(
                encoding="utf-8"
            )
            self.assertEqual(summary, "- source data root: `<redacted>`\n")

    def test_package_artifact_includes_smoke_outputs_and_reproducible_zip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result_root = root / "result"
            plots_dir = result_root / "plots"
            plots_dir.mkdir(parents=True)
            (result_root / "summary.json").write_text(
                json.dumps({"result_root": "/home/user/private/run", "num_joints": 23}),
                encoding="utf-8",
            )
            (result_root / "filter_metrics.csv").write_text(
                "method,run_dir,filter_error_deg\nfilter,/home/user/private/run,4.2\n",
                encoding="utf-8",
            )
            (plots_dir / "filter_vs_baselines.svg").write_text("<svg>ok</svg>", encoding="utf-8")
            (result_root / "private_motion.npz").write_bytes(b"raw data must not be copied")

            output_dir = root / "paper"
            output_zip = root / "paper.zip"
            manifest = package_artifact(result_root, output_dir, output_zip=output_zip)
            first_zip_bytes = output_zip.read_bytes()
            second_manifest = package_artifact(result_root, output_dir, output_zip=output_zip)

            self.assertEqual(manifest, second_manifest)
            self.assertEqual(output_zip.read_bytes(), first_zip_bytes)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertIn("summary.json", manifest["files"])
            self.assertIn("filter_metrics.csv", manifest["files"])
            self.assertIn("plots/filter_vs_baselines.svg", manifest["files"])
            self.assertNotIn("private_motion.npz", manifest["files"])

            copied_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(copied_summary["result_root"], "<redacted>")
            copied_metrics = (output_dir / "filter_metrics.csv").read_text(encoding="utf-8")
            self.assertNotIn("run_dir", copied_metrics)
            self.assertNotIn("/home/user/private/run", copied_metrics)
            self.assertTrue(all(record["sha256"] for record in manifest["file_records"]))


if __name__ == "__main__":
    unittest.main()
