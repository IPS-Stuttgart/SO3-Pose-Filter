from __future__ import annotations

import tomllib
import unittest

from _path import ROOT


class PackagingMetadataTests(unittest.TestCase):
    def test_project_metadata_declares_citation_license_and_urls(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = pyproject["project"]

        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(project["license"], {"file": "LICENSE"})
        self.assertIn("Repository", project["urls"])
        self.assertIn("SO(3)", project["keywords"])
        self.assertTrue((ROOT / "LICENSE").exists())
        self.assertTrue((ROOT / "CITATION.cff").exists())
        self.assertTrue((ROOT / "MANIFEST.in").exists())
        self.assertTrue((ROOT / "requirements-dev.txt").exists())

    def test_reproduce_doc_contains_smoke_manifest_and_package_commands(self) -> None:
        text = (ROOT / "docs" / "reproduce.md").read_text(encoding="utf-8")
        expected_fragments = [
            "scripts/make_toy_amass.py --output data/tiny_amass --sequences 6 --frames 80",
            "scripts/run_experiment.py --config configs/example.json",
            "scripts/write_experiment_manifest.py",
            "scripts/package_paper_artifact.py",
            "--output-zip results/toy-smoke-public.zip",
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
