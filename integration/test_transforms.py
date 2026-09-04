import tempfile
import unittest
from pathlib import Path

import numpy as np

from integration.transforms import (
    BOX_SYMMETRIES,
    box_disagreement,
    compose_world_box_poses,
    evaluate_camera_pose,
    foundationpose_pose_last_from_original,
    fuse_world_poses,
    save_latest_transforms,
    tracker_root_and_camera,
    vive_alignment_candidate,
)


def T(R=None, t=(0.0, 0.0, 0.0)):
    out = np.eye(4)
    if R is not None:
        out[:3, :3] = R
    out[:3, 3] = t
    return out


class TransformTests(unittest.TestCase):
    def test_external_world_pose_is_independent(self):
        E_T_box = T(t=(0.2, -0.1, 1.0))
        ext, g1, camera = compose_world_box_poses(
            np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)),
            np.zeros((4, 4)), E_T_box, np.eye(4),
        )
        np.testing.assert_allclose(ext, E_T_box)
        self.assertIsNone(g1)
        self.assertIsNone(camera)

    def test_g1_world_chain(self):
        E_T_V = T(t=(1, 0, 0))
        V_T_T = T(t=(0, 2, 0))
        T_T_B = T(t=(0, 0, 3))
        B_T_C = T(t=(4, 0, 0))
        C_T_box = T(t=(0, 5, 0))
        ext, g1, E_T_C = compose_world_box_poses(
            E_T_V, V_T_T, T_T_B, B_T_C, None, C_T_box
        )
        self.assertIsNone(ext)
        np.testing.assert_allclose(E_T_C, E_T_V @ V_T_T @ T_T_B @ B_T_C)
        np.testing.assert_allclose(g1, E_T_C @ C_T_box)

    def test_external_only_fusion(self):
        ext = T(t=(1, 2, 3))
        result = fuse_world_poses(ext, True, 0.8, None, False, 0.0)
        self.assertEqual(result.source, "EXTERNAL")
        np.testing.assert_allclose(result.pose, ext)

    def test_g1_only_fusion(self):
        g1 = T(t=(1, 2, 3))
        result = fuse_world_poses(None, False, 0.0, g1, True, 0.7)
        self.assertEqual(result.source, "G1")
        np.testing.assert_allclose(result.pose, g1)

    def test_both_camera_weighted_fusion(self):
        ext = T(t=(0, 0, 1))
        g1 = T(t=(0.3, 0, 1))
        result = fuse_world_poses(ext, True, 0.75, g1, True, 0.25)
        self.assertEqual(result.source, "BOTH")
        np.testing.assert_allclose(result.pose[:3, 3], [0.075, 0, 1])

    def test_neither_holds_for_only_025_seconds(self):
        last = T(t=(1, 2, 3))
        held = fuse_world_poses(
            None, False, 0, None, False, 0,
            last_pose=last, last_pose_time=10.0, now=10.25,
        )
        self.assertTrue(held.valid)
        self.assertTrue(held.held)
        self.assertEqual(held.source, "NONE")
        expired = fuse_world_poses(
            None, False, 0, None, False, 0,
            last_pose=last, last_pose_time=10.0, now=10.251,
        )
        self.assertFalse(expired.valid)
        self.assertIsNone(expired.pose)

    def test_raw_and_cube_symmetry_errors(self):
        a = T()
        b = T(BOX_SYMMETRIES[1], (0.003, 0.004, 0.0))
        error = box_disagreement(a, b)
        self.assertAlmostEqual(error.translation_mm, 5.0)
        self.assertAlmostEqual(error.rotation_raw_deg, 90.0)
        self.assertAlmostEqual(error.rotation_symmetry_deg, 0.0, places=5)

    def test_cube_group_has_twenty_four_proper_unique_rotations(self):
        self.assertEqual(len(BOX_SYMMETRIES), 24)
        for rotation in BOX_SYMMETRIES:
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
        rounded = {tuple(np.round(rotation, 10).flat) for rotation in BOX_SYMMETRIES}
        self.assertEqual(len(rounded), 24)

    def test_24_way_cube_symmetry_rotation_fusion(self):
        ext = T()
        for symmetry in BOX_SYMMETRIES:
            result = fuse_world_poses(ext, True, 0.5, T(symmetry), True, 0.5)
            np.testing.assert_allclose(result.pose[:3, :3], np.eye(3), atol=1e-7)

    def test_quaternion_rotation_fusion_midpoint(self):
        angle = np.deg2rad(20.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        half = np.deg2rad(10.0)
        expected = np.array([
            [np.cos(half), -np.sin(half), 0],
            [np.sin(half), np.cos(half), 0],
            [0, 0, 1],
        ])
        result = fuse_world_poses(T(), True, 0.5, T(rotation), True, 0.5)
        np.testing.assert_allclose(result.pose[:3, :3], expected, atol=1e-7)

    def test_pose_validity_and_stale_rejection(self):
        K = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]])
        pose = T(t=(0, 0, 1.0))
        depth = np.ones((480, 640), dtype=float)
        valid = evaluate_camera_pose(pose, 9.9, 10.0, depth, K, (480, 640))
        self.assertTrue(valid.valid, valid.reason)
        self.assertGreater(valid.quality, 0.5)
        older = evaluate_camera_pose(pose, 9.5, 10.0, depth, K, (480, 640))
        self.assertTrue(older.valid, older.reason)
        self.assertLess(older.quality, valid.quality)
        stale = evaluate_camera_pose(pose, 9.0, 10.0, depth, K, (480, 640))
        self.assertFalse(stale.valid)
        self.assertEqual(stale.reason, "stale")
        inconsistent = evaluate_camera_pose(
            pose, 9.9, 10.0, np.full((480, 640), 3.0), K, (480, 640)
        )
        self.assertFalse(inconsistent.valid)
        self.assertEqual(inconsistent.reason, "depth mismatch")
        behind = evaluate_camera_pose(T(t=(0, 0, -1)), 9.9, 10.0, depth, K, (480, 640))
        self.assertFalse(behind.valid)
        self.assertEqual(behind.reason, "implausible Z")
        offscreen = evaluate_camera_pose(T(t=(5, 0, 1)), 9.9, 10.0, depth, K, (480, 640))
        self.assertFalse(offscreen.valid)
        self.assertEqual(offscreen.reason, "outside image")

    def test_foundationpose_reseed_uses_centered_pose_last_convention(self):
        camera_T_original = T(t=(0.1, -0.2, 1.3))
        center_from_original = T(t=(-0.02, 0.03, -0.04))
        pose_last = foundationpose_pose_last_from_original(
            camera_T_original, center_from_original
        )
        np.testing.assert_allclose(
            pose_last @ center_from_original, camera_T_original, atol=1e-12
        )

    def test_vive_alignment_and_runtime_chain_are_inverse(self):
        E_T_B = T(t=(1.0, 2.0, 3.0))
        T_T_B = T(t=(0.06, 0.0, 0.08))
        V_T_T = T(t=(0.4, -0.2, 0.1))
        B_T_C = T(t=(0.0, 0.0, 0.43))
        E_T_V = vive_alignment_candidate(E_T_B, T_T_B, V_T_T)
        actual_B, E_T_C, E_T_T = tracker_root_and_camera(E_T_V, V_T_T, T_T_B, B_T_C)
        np.testing.assert_allclose(actual_B, E_T_B, atol=1e-12)
        np.testing.assert_allclose(E_T_C, actual_B @ B_T_C, atol=1e-12)
        np.testing.assert_allclose(E_T_T, E_T_V @ V_T_T, atol=1e-12)

    def test_save_only_available_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savetxt(path / "C_T_box.txt", np.eye(4))
            np.savetxt(path / "E_T_box_g1.txt", np.eye(4))
            save_latest_transforms(
                path, E_T_box=np.eye(4), C_T_box=None,
                E_T_box_ext=np.eye(4), E_T_box_g1=None,
                E_T_box_fused=np.eye(4),
            )
            self.assertEqual(
                sorted(p.name for p in path.iterdir()),
                ["E_T_box.txt", "E_T_box_ext.txt", "E_T_box_fused.txt"],
            )


if __name__ == "__main__":
    unittest.main()
