import tempfile
import unittest
from pathlib import Path

import numpy as np

from integration.transforms import (
    BOX_SYMMETRIES,
    box_disagreement,
    compose_box_poses,
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
    def test_exact_independent_composition(self):
        K_T_E = T(t=(1, 0, 0))
        E_T_box = T(t=(0, 2, 0))
        K_T_C = T(t=(0, 0, 3))
        C_T_box = T(t=(4, 0, 0))
        ext, g1 = compose_box_poses(K_T_E, E_T_box, K_T_C, C_T_box)
        np.testing.assert_allclose(ext, K_T_E @ E_T_box)
        np.testing.assert_allclose(g1, K_T_C @ C_T_box)

    def test_all_availability_states(self):
        I = np.eye(4)
        expected = [(False, False), (True, False), (False, True), (True, True)]
        inputs = [
            (None, None, None, None),
            (I, I, None, None),
            (None, None, I, I),
            (I, I, I, I),
        ]
        for values, availability in zip(inputs, expected):
            actual = compose_box_poses(*values)
            self.assertEqual(tuple(x is not None for x in actual), availability)

    def test_raw_and_cuboid_symmetry_errors(self):
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

    def test_vive_alignment_and_runtime_chain_are_inverse(self):
        K_T_B_vision = T(t=(1.0, 2.0, 3.0))
        T_T_B = T(t=(0.06, 0.0, 0.08))
        V_T_T = T(t=(0.4, -0.2, 0.1))
        B_T_C = T(t=(0.0, 0.0, 0.43))
        K_T_V = vive_alignment_candidate(K_T_B_vision, T_T_B, V_T_T)
        K_T_B, K_T_C, K_T_T = tracker_root_and_camera(
            K_T_V, V_T_T, T_T_B, B_T_C
        )
        np.testing.assert_allclose(K_T_B, K_T_B_vision, atol=1e-12)
        np.testing.assert_allclose(K_T_C, K_T_B @ B_T_C, atol=1e-12)
        np.testing.assert_allclose(K_T_T, K_T_V @ V_T_T, atol=1e-12)

    def test_save_only_available_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_latest_transforms(
                path, E_T_box=np.eye(4), C_T_box=None,
                K_T_box_ext=np.eye(4), K_T_box_g1=None,
            )
            self.assertEqual(
                sorted(p.name for p in path.iterdir()),
                ["E_T_box.txt", "K_T_box_ext.txt"],
            )


if __name__ == "__main__":
    unittest.main()
