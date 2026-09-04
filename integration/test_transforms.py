import tempfile
import unittest
from pathlib import Path

import numpy as np

from integration.transforms import (
    BOX_SYMMETRIES,
    LOST,
    PARTIAL,
    TRACKING,
    box_disagreement,
    compose_world_box_poses,
    evaluate_camera_pose,
    foundationpose_pose_last_from_original,
    fuse_world_poses,
    render_expected_cube_depth,
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

    def test_g1_lost_external_tracking_uses_external_only(self):
        ext = T(t=(1, 2, 3))
        result = fuse_world_poses(ext, True, 0.8, None, False, 0.0)
        self.assertEqual(result.source, "EXTERNAL")
        np.testing.assert_allclose(result.pose, ext)

    def test_external_lost_g1_tracking_uses_g1_only(self):
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

    def test_both_lost_has_no_current_or_held_output(self):
        result = fuse_world_poses(None, False, 0, None, False, 0)
        self.assertFalse(result.valid)
        self.assertEqual(result.source, "NONE")
        self.assertIsNone(result.pose)

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

    @staticmethod
    def _camera_test_data(pose):
        K = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]])
        rendered = render_expected_cube_depth(pose, K, (480, 640))
        depth = np.zeros((480, 640), dtype=float)
        surface = np.isfinite(rendered.depth_m)
        depth[surface] = rendered.depth_m[surface]
        return K, depth, surface

    def test_fully_visible_cube_is_tracking(self):
        pose = T(t=(0, 0, 1.0))
        K, depth, _ = self._camera_test_data(pose)
        result = evaluate_camera_pose(pose, 9.9, 10.0, depth, K, depth.shape)
        self.assertEqual(result.state, TRACKING, result.reason)
        self.assertGreater(result.surface_agreement, 0.99)
        self.assertGreater(result.quality, 0.5)

    def test_partially_outside_image_is_partial_with_support(self):
        pose = T(t=(1.0, 0, 1.0))
        K, depth, _ = self._camera_test_data(pose)
        result = evaluate_camera_pose(pose, 9.9, 10.0, depth, K, depth.shape)
        self.assertEqual(result.state, PARTIAL, result.reason)
        self.assertGreater(result.image_overlap, 0.20)
        self.assertLess(result.image_overlap, 0.70)

    def test_partially_occluded_cube_is_partial(self):
        pose = T(t=(0, 0, 1.0))
        K, depth, surface = self._camera_test_data(pose)
        ys, xs = np.where(surface)
        occluded = xs < np.median(xs)
        depth[ys[occluded], xs[occluded]] -= 0.10
        result = evaluate_camera_pose(pose, 9.9, 10.0, depth, K, depth.shape)
        self.assertEqual(result.state, PARTIAL, result.reason)
        self.assertGreater(result.occlusion_ratio, 0.45)
        self.assertGreater(result.surface_agreement, 0.45)

    def test_completely_occluded_cube_is_lost(self):
        pose = T(t=(0, 0, 1.0))
        K, depth, surface = self._camera_test_data(pose)
        depth[surface] -= 0.10
        result = evaluate_camera_pose(pose, 9.9, 10.0, depth, K, depth.shape)
        self.assertEqual(result.state, LOST)
        self.assertGreater(result.occlusion_ratio, 0.99)
        self.assertEqual(result.surface_agreement, 0.0)

    def test_completely_outside_image_is_lost(self):
        pose = T(t=(5, 0, 1.0))
        K = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]])
        depth = np.ones((480, 640), dtype=float)
        result = evaluate_camera_pose(pose, 9.9, 10.0, depth, K, depth.shape)
        self.assertEqual(result.state, LOST)
        self.assertEqual(result.reason, "outside image")

    def test_unrelated_background_depth_is_lost(self):
        pose = T(t=(0, 0, 1.0))
        K, depth, surface = self._camera_test_data(pose)
        depth[surface] = 3.0
        result = evaluate_camera_pose(pose, 9.9, 10.0, depth, K, depth.shape)
        self.assertEqual(result.state, LOST)
        self.assertEqual(result.reason, "surface depth mismatch")
        self.assertGreater(result.behind_ratio, 0.99)

    def test_no_depth_support_is_lost(self):
        pose = T(t=(0, 0, 1.0))
        K, depth, _ = self._camera_test_data(pose)
        depth[:] = 0.0
        result = evaluate_camera_pose(pose, 9.9, 10.0, depth, K, depth.shape)
        self.assertEqual(result.state, LOST)
        self.assertEqual(result.reason, "insufficient depth")
        self.assertGreater(result.missing_depth_ratio, 0.99)

    def test_stale_pose_is_lost_even_with_surface_support(self):
        pose = T(t=(0, 0, 1.0))
        K, depth, _ = self._camera_test_data(pose)
        result = evaluate_camera_pose(pose, 9.0, 10.0, depth, K, depth.shape)
        self.assertEqual(result.state, LOST)
        self.assertEqual(result.reason, "stale")
        self.assertEqual(result.quality, 0.0)

    def test_implausible_camera_z_is_lost(self):
        pose = T(t=(0, 0, -1.0))
        K = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]])
        depth = np.ones((480, 640), dtype=float)
        result = evaluate_camera_pose(pose, 9.9, 10.0, depth, K, depth.shape)
        self.assertEqual(result.state, LOST)
        self.assertEqual(result.reason, "implausible Z")

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
