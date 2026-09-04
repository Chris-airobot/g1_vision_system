"""Independent, latest-frame-only FoundationPose worker."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from integration.transforms import validate_transform


class FoundationPoseWorker:
    """Own one complete estimator and never block its caller.

    A queue of size one drops superseded camera frames. Each instance imports
    and constructs its own FoundationPose estimator, scorer, refiner, tracking
    state, and rasterizer context. No estimator object or pose state is shared.
    """

    def __init__(
        self,
        name: str,
        foundationpose_root: Path,
        mesh_path: Path,
        init_dir: Path,
        output_dir: Path,
        track_iterations: int = 1,
        register_iterations: int = 5,
    ) -> None:
        self.name = name
        self.foundationpose_root = Path(foundationpose_root)
        self.mesh_path = Path(mesh_path)
        self.init_dir = Path(init_dir)
        self.output_dir = Path(output_dir)
        self.track_iterations = track_iterations
        self.register_iterations = register_iterations
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._pose: Optional[np.ndarray] = None
        self._pose_time = 0.0
        self._status = "STARTING"
        self._error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"fp-{name}")
        self._thread.start()

    def submit(self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> None:
        item = (rgb.copy(), depth_m.copy(), np.asarray(K, dtype=float).copy())
        try:
            self._frames.put_nowait(item)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(item)
            except queue.Full:
                pass

    def get(self) -> tuple[Optional[np.ndarray], float, str, str]:
        with self._lock:
            return (
                None if self._pose is None else self._pose.copy(),
                self._pose_time,
                self._status,
                self._error,
            )

    def stop(self) -> None:
        self._stop.set()

    def _set_status(self, status: str, error: str = "") -> None:
        with self._lock:
            self._status = status
            self._error = error

    @staticmethod
    def _load_depth(path: Path) -> np.ndarray:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"cannot read initialization depth: {path}")
        depth = raw.astype(np.float32) * 0.001
        depth[(depth < 0.001) | (depth > 10.0)] = 0.0
        return depth

    def _run(self) -> None:
        try:
            # FoundationPose has repository-local absolute imports.
            sys.path.insert(0, str(self.foundationpose_root))
            from estimater import (  # pylint: disable=import-outside-toplevel
                FoundationPose,
                PoseRefinePredictor,
                ScorePredictor,
                dr,
                set_logging_format,
                set_seed,
                trimesh,
            )

            self.output_dir.mkdir(parents=True, exist_ok=True)
            rgb_path = self.init_dir / "rgb" / "000000.png"
            depth_path = self.init_dir / "depth" / "000000.png"
            mask_path = self.init_dir / "masks" / "000000.png"
            K_path = self.init_dir / "cam_K.txt"
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if bgr is None or mask_raw is None or not K_path.exists():
                raise RuntimeError(
                    f"incomplete init directory {self.init_dir}; expected cam_K.txt and "
                    "rgb/depth/masks/000000.png"
                )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = self._load_depth(depth_path)
            mask = mask_raw.astype(bool)
            K = np.loadtxt(K_path).reshape(3, 3)

            set_logging_format()
            set_seed(0)
            mesh = trimesh.load(self.mesh_path, force="mesh")
            estimator = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=ScorePredictor(),
                refiner=PoseRefinePredictor(),
                debug_dir=str(self.output_dir),
                debug=0,
                glctx=dr.RasterizeCudaContext(),
            )
            pose = estimator.register(
                K=K, rgb=rgb, depth=depth, ob_mask=mask,
                iteration=self.register_iterations,
            )
            # Registration seeds this worker's private tracker. Do not publish
            # the prerecorded init pose as though it came from the live camera.
            with self._lock:
                self._status = "READY"
            np.savetxt(self.output_dir / "initial_pose.txt", pose)

            while not self._stop.is_set():
                try:
                    rgb, depth, K = self._frames.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    pose = estimator.track_one(
                        rgb=rgb, depth=depth, K=K, iteration=self.track_iterations
                    )
                    pose = validate_transform(np.asarray(pose, dtype=float), f"{self.name}_T_box")
                    with self._lock:
                        self._pose = pose
                        self._pose_time = time.monotonic()
                        self._status = "TRACKING"
                        self._error = ""
                except Exception as exc:  # a bad frame must not kill the worker
                    self._set_status("FRAME ERROR", repr(exc))
        except Exception as exc:  # missing init/GPU/dependency is a safe disabled state
            self._set_status("DISABLED", repr(exc))
