from __future__ import annotations

import unittest

from _path import SCRIPTS  # noqa: F401
from download_accad_webdav_sample import select_pose_npz_files


class AccadWebdavSampleTests(unittest.TestCase):
    def test_select_pose_npz_files_prefers_pose_files(self) -> None:
        paths = [
            "subject/readme.txt",
            "subject/neutral.npz",
            "Male2Walking_c3d/B9 - Walk turn left 90_poses.npz",
            "Female1General_c3d/A1 - Stand_poses.npz",
        ]

        selected = select_pose_npz_files(paths, max_files=1, candidate_limit=3)

        self.assertEqual(
            selected,
            [
                "Female1General_c3d/A1 - Stand_poses.npz",
                "Male2Walking_c3d/B9 - Walk turn left 90_poses.npz",
                "subject/neutral.npz",
            ],
        )

    def test_select_pose_npz_files_requires_candidate_budget(self) -> None:
        with self.assertRaises(ValueError):
            select_pose_npz_files(["seq_poses.npz"], max_files=2, candidate_limit=1)


if __name__ == "__main__":
    unittest.main()
