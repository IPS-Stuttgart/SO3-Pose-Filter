from __future__ import annotations

import unittest

from _path import SCRIPTS  # noqa: F401
from download_webdav_amass_sample import select_amass_npz_files


class WebdavAmassSampleTests(unittest.TestCase):
    def test_select_amass_npz_files_prefers_pose_files(self) -> None:
        paths = [
            "subject/readme.txt",
            "subject/neutral.npz",
            "Male2Walking_c3d/B9 - Walk turn left 90_poses.npz",
            "Female1General_c3d/A1 - Stand_poses.npz",
        ]

        selected = select_amass_npz_files(paths, max_files=1, candidate_limit=3)

        self.assertEqual(
            selected,
            [
                "Female1General_c3d/A1 - Stand_poses.npz",
                "Male2Walking_c3d/B9 - Walk turn left 90_poses.npz",
                "subject/neutral.npz",
            ],
        )

    def test_select_amass_npz_files_accepts_dataset_independent_paths(self) -> None:
        paths = [
            "KIT/3/bend_left01_poses.npz",
            "ACCAD/Male1Walking_c3d/B1 - walk_poses.npz",
            "KIT/3/bend_left01_stageii.npz",
        ]

        selected = select_amass_npz_files(paths, max_files=2, candidate_limit=3)

        self.assertEqual(
            selected,
            [
                "ACCAD/Male1Walking_c3d/B1 - walk_poses.npz",
                "KIT/3/bend_left01_poses.npz",
                "KIT/3/bend_left01_stageii.npz",
            ],
        )

    def test_select_amass_npz_files_requires_candidate_budget(self) -> None:
        with self.assertRaises(ValueError):
            select_amass_npz_files(["seq_poses.npz"], max_files=2, candidate_limit=1)


if __name__ == "__main__":
    unittest.main()
