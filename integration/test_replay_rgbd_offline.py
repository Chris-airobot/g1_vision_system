import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from integration.replay_rgbd_offline import (
    RECORDER_COLUMNS,
    camera_row_values,
    depth_png_to_metres,
    read_recording_rows,
    select_rows,
)


class OfflineReplayHelperTests(unittest.TestCase):
    def test_exact_recorder_schema_and_frames_csv_order(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory)
            with (recording / "frames.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=RECORDER_COLUMNS)
                writer.writeheader()
                for frame in (7, 3, 11):
                    row = {field: "" for field in RECORDER_COLUMNS}
                    row.update({
                        "frame": str(frame),
                        "g1_rgb": f"g1/rgb/{frame:06d}.jpg",
                        "g1_depth": f"g1/depth/{frame:06d}.png",
                        "g1_timestamp": str(100.0 + frame),
                    })
                    writer.writerow(row)
            rows = read_recording_rows(recording)
            self.assertEqual([row["frame"] for row in rows], ["7", "3", "11"])
            self.assertEqual(
                camera_row_values(rows[0], "g1"),
                ("g1/rgb/000007.jpg", "g1/depth/000007.png", "107.0"),
            )

    def test_start_end_are_order_indices_and_end_is_exclusive(self):
        rows = [{"frame": str(index)} for index in range(10)]
        selected = select_rows(rows, start=1, end=8, stride=3)
        self.assertEqual([row["frame"] for row in selected], ["1", "4", "7"])

    def test_uint16_millimetres_convert_to_filtered_float32_metres(self):
        raw = np.array([[0, 1, 1250, 10000, 10001]], dtype=np.uint16)
        depth = depth_png_to_metres(raw)
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_allclose(depth, [[0.0, 0.001, 1.25, 10.0, 0.0]])

    def test_bad_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            select_rows([], start=-1, end=None, stride=1)
        with self.assertRaises(ValueError):
            select_rows([], start=0, end=None, stride=0)


if __name__ == "__main__":
    unittest.main()
