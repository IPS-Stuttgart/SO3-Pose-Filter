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


if __name__ == "__main__":
    unittest.main()
