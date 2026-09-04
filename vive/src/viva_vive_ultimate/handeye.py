"""VIVE Ultimate Tracker + D435i + ChArUco 联合标定核心。

本模块不接触硬件，只处理图案检测、SE(3)、手眼求解、结果序列化，因此可以用合成数据
完整测试。原先配套的 D435i 采集/求解 CLI（PU-200A 支架方案）已移除；现在的使用者是
``g1_tracker_system``：用 :class:`CharucoEstimator` 做板检测，用 :func:`solve_handeye`
把 ``B_T_K = B_T_C @ C_T_K``（机器人 FK + 板载相机）当作 ``C_T_K`` 求 tracker 到 root 的 ``T_T_B``。

统一变换记法
------------
``A_T_B`` 表示 :math:`{}^A T_B`，即把 B 系坐标变换到 A 系。手眼数据固定为：

``V_T_T``
    重映射后的 tracker 机体系 T 到本次 VIVE 地图 V。V 已经是 Z-up。
``C_T_K``
    ChArUco 原生板坐标 K 到 D435i color optical frame C。它就是 solvePnP 输出，
    **不取逆**。
``T_T_C``
    待求固定外参，D435i color optical frame C 到 tracker 机体系 T。

标定板固定时每个样本满足 ``V_T_T @ T_T_C @ C_T_K = V_T_K``。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

FORMAT_VERSION = 1
DEFAULT_BODY_AXES = "Y,-X,Z"


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - 依赖错误只在部署机触发
        raise ImportError('联合标定需要 numpy：pip install -e ".[calibration]"') from exc
    return np


def _rotation():
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:  # pragma: no cover
        raise ImportError('联合标定需要 scipy：pip install -e ".[calibration]"') from exc
    return Rotation


def identity_transform():
    return _np().eye(4, dtype=float)


def make_transform(R, t):
    """从 3x3 旋转和三维平移构造齐次变换。"""
    np = _np()
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_transform(T):
    np = _np()
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    return make_transform(R.T, -R.T @ T[:3, 3])


def transform_from_quat(pos, quat_xyzw):
    Rotation = _rotation()
    return make_transform(Rotation.from_quat(quat_xyzw).as_matrix(), pos)


def transform_to_pose(T) -> tuple[list[float], list[float]]:
    """返回 ``(translation, quaternion_xyzw)``。"""
    np = _np()
    Rotation = _rotation()
    T = np.asarray(T, dtype=float)
    return T[:3, 3].tolist(), Rotation.from_matrix(T[:3, :3]).as_quat().tolist()


def transform_angle_deg(T) -> float:
    Rotation = _rotation()
    return float(math.degrees(Rotation.from_matrix(T[:3, :3]).magnitude()))


def transform_distance(a, b) -> tuple[float, float]:
    """两个位姿的差，返回 ``(平移米, 旋转度)``。"""
    d = invert_transform(a) @ b
    return float(_np().linalg.norm(d[:3, 3])), transform_angle_deg(d)


def average_transforms(transforms: Sequence, weights: Sequence[float] | None = None):
    """在 SO(3) 上平均旋转、在 R³ 平均平移。"""
    np = _np()
    Rotation = _rotation()
    Ts = np.asarray(transforms, dtype=float)
    if Ts.ndim != 3 or Ts.shape[1:] != (4, 4) or len(Ts) == 0:
        raise ValueError("transforms 需要形如 (N,4,4) 且非空")
    w = None if weights is None else np.asarray(weights, dtype=float)
    if w is not None:
        if w.shape != (len(Ts),) or not np.all(w >= 0) or w.sum() <= 0:
            raise ValueError("weights 需要是非负 (N,) 且至少一个非零")
        t = np.average(Ts[:, :3, 3], axis=0, weights=w)
    else:
        t = Ts[:, :3, 3].mean(axis=0)
    R = Rotation.from_matrix(Ts[:, :3, :3]).mean(weights=w).as_matrix()
    return make_transform(R, t)


def interpolate_transform(t_ns: int, t0_ns: int, T0, t1_ns: int, T1):
    """位置线性、姿态 SLERP。时间必须在两帧之间。"""
    np = _np()
    Rotation = _rotation()
    from scipy.spatial.transform import Slerp

    if t1_ns <= t0_ns:
        raise ValueError("插值两帧时间必须递增")
    if t_ns < t0_ns or t_ns > t1_ns:
        raise ValueError("拒绝外推 tracker pose")
    u = (t_ns - t0_ns) / (t1_ns - t0_ns)
    p = (1.0 - u) * np.asarray(T0)[:3, 3] + u * np.asarray(T1)[:3, 3]
    rots = Rotation.from_matrix(np.stack([np.asarray(T0)[:3, :3],
                                          np.asarray(T1)[:3, :3]]))
    R = Slerp([0.0, 1.0], rots)([u]).as_matrix()[0]
    return make_transform(R, p)


def _matrix_list(T) -> list[list[float]]:
    return _np().asarray(T, dtype=float).tolist()


def _validate_transform(T, name: str):
    np = _np()
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name} 不是有限的 4x4 矩阵")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} 最后一行不是 [0,0,0,1]")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or abs(np.linalg.det(R) - 1) > 1e-5:
        raise ValueError(f"{name} 的旋转部分不是右手正交矩阵")
    return T


@dataclass(frozen=True)
class BoardConfig:
    dictionary: str = "DICT_6X6_250"
    squares_x: int = 7
    squares_y: int = 7
    square_length_m: float = 0.04
    marker_length_m: float = 0.03
    legacy_pattern: bool = False
    # 默认世界原点：板中心，X 向纸面右、Y 向纸面上、Z 从打印面向外。
    O_T_K: list[list[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0, -0.14],
        [0.0, -1.0, 0.0, 0.14],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    def validate(self) -> "BoardConfig":
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError("ChArUco 至少需要 2x2 squares")
        if not (0 < self.marker_length_m < self.square_length_m):
            raise ValueError("marker_length_m 必须大于 0 且小于 square_length_m")
        _validate_transform(self.O_T_K, "O_T_K")
        return self

    @property
    def outer_size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_length_m, self.squares_y * self.square_length_m

    @property
    def digest(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def opencv_board(self):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError('需要 opencv-contrib-python：pip install -e ".[calibration]"') from exc
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"OpenCV 不认识字典 {self.dictionary!r}")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_length_m,
            self.marker_length_m, dictionary)
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(self.legacy_pattern)
        return board, dictionary


def load_board_config(path: str | Path) -> BoardConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError('读取 board YAML 需要 PyYAML：pip install -e ".[calibration]"') from exc
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    # 兼容更易读的键名；文件里矩阵优先。
    matrix = data.pop("world_from_board", data.pop("O_T_K", None))
    sx = int(data.get("squares_x", 7))
    sy = int(data.get("squares_y", 7))
    length = float(data.get("square_length_m", 0.04))
    if matrix is None:
        matrix = [[1, 0, 0, -sx * length / 2],
                  [0, -1, 0, sy * length / 2],
                  [0, 0, -1, 0], [0, 0, 0, 1]]
    cfg = BoardConfig(
        dictionary=str(data.get("dictionary", "DICT_6X6_250")),
        squares_x=sx, squares_y=sy, square_length_m=length,
        marker_length_m=float(data.get("marker_length_m", 0.03)),
        legacy_pattern=bool(data.get("legacy_pattern", False)),
        O_T_K=[[float(v) for v in row] for row in matrix],
    )
    return cfg.validate()


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    camera_matrix: list[list[float]]
    distortion: list[float]
    serial: str = ""
    model: str = "opencv_brown5"
    source: str = "factory"
    rms_px: float | None = None
    views: int | None = None
    created: str = ""

    @property
    def K(self):
        return _np().asarray(self.camera_matrix, dtype=float)

    @property
    def D(self):
        return _np().asarray(self.distortion, dtype=float)

    def validate(self) -> "CameraIntrinsics":
        np = _np()
        K = self.K
        if self.width <= 0 or self.height <= 0 or K.shape != (3, 3):
            raise ValueError("相机内参尺寸或 camera_matrix 不合法")
        if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("相机焦距不合法")
        if not np.isfinite(self.D).all():
            raise ValueError("畸变系数含非有限值")
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({"format": FORMAT_VERSION, "kind": "d435i_color_intrinsics"})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CameraIntrinsics":
        if d.get("kind") != "d435i_color_intrinsics":
            raise ValueError("不是 D435i color 内参文件")
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**fields).validate()


def save_intrinsics(path: str | Path, intr: CameraIntrinsics) -> None:
    Path(path).write_text(json.dumps(intr.to_dict(), ensure_ascii=False, indent=2),
                          encoding="utf-8")


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    return CameraIntrinsics.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class CharucoDetection:
    corners: list[list[float]]
    ids: list[int]
    C_T_K: list[list[float]] | None
    reprojection_rms_px: float | None
    marker_count: int

    @property
    def corner_count(self) -> int:
        return len(self.ids)


class CharucoEstimator:
    """固定图案的检测器；可反复用于实时帧。"""

    def __init__(self, config: BoardConfig, min_corners: int = 12) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError('需要 opencv-contrib-python：pip install -e ".[calibration]"') from exc
        self.cv2 = cv2
        self.config = config.validate()
        self.board, dictionary = config.opencv_board()
        self.min_corners = int(min_corners)
        if hasattr(cv2.aruco, "CharucoDetector"):
            self.detector = cv2.aruco.CharucoDetector(self.board)
            self.aruco_detector = None
        else:  # pragma: no cover - 老 OpenCV 兼容
            self.detector = None
            self.aruco_detector = cv2.aruco.ArucoDetector(
                dictionary, cv2.aruco.DetectorParameters())

    def detect(self, image, intrinsics: CameraIntrinsics | None = None) -> CharucoDetection | None:
        cv2, np = self.cv2, _np()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if self.detector is not None:
            cc, ci, mc, mi = self.detector.detectBoard(gray)
        else:  # pragma: no cover
            mc, mi, _ = self.aruco_detector.detectMarkers(gray)
            if mi is None:
                return None
            _, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, self.board)
        marker_count = 0 if mi is None else len(mi)
        if ci is None or cc is None or len(ci) < self.min_corners:
            return None
        ids = np.asarray(ci, dtype=int).reshape(-1)
        pts = np.asarray(cc, dtype=float).reshape(-1, 2)
        C_T_K = None
        rms = None
        if intrinsics is not None:
            if (gray.shape[1], gray.shape[0]) != (intrinsics.width, intrinsics.height):
                raise ValueError("图像分辨率与相机内参不一致")
            obj = np.asarray(self.board.getChessboardCorners(), dtype=np.float64)[ids]
            ok, rvec, tvec = cv2.solvePnP(
                obj, pts.astype(np.float64), intrinsics.K, intrinsics.D,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or float(tvec[2]) <= 0:
                return None
            R, _ = cv2.Rodrigues(rvec)
            C_T_K_np = make_transform(R, tvec.reshape(3))
            projected, _ = cv2.projectPoints(obj, rvec, tvec, intrinsics.K, intrinsics.D)
            err = projected.reshape(-1, 2) - pts
            rms = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
            C_T_K = _matrix_list(C_T_K_np)
        return CharucoDetection(pts.tolist(), ids.tolist(), C_T_K, rms, marker_count)

    def draw(self, image, detection: CharucoDetection | None,
             intrinsics: CameraIntrinsics | None = None):
        cv2, np = self.cv2, _np()
        out = image.copy()
        if detection is None:
            return out
        corners = np.asarray(detection.corners, dtype=np.float32).reshape(-1, 1, 2)
        ids = np.asarray(detection.ids, dtype=np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedCornersCharuco(out, corners, ids, (0, 255, 0))
        if detection.C_T_K is not None and intrinsics is not None:
            T = np.asarray(detection.C_T_K)
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            cv2.drawFrameAxes(out, intrinsics.K, intrinsics.D, rvec, T[:3, 3],
                              self.config.square_length_m * 2)
        return out


def calibrate_camera(detections: Sequence[CharucoDetection], config: BoardConfig,
                     width: int, height: int, initial: CameraIntrinsics | None = None,
                     serial: str = "") -> tuple[CameraIntrinsics, list[float]]:
    """从多视角 ChArUco 角点标定 color 相机内参。"""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError('需要 opencv-contrib-python：pip install -e ".[calibration]"') from exc
    np = _np()
    board, _ = config.opencv_board()
    all_obj = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    obj, img = [], []
    for d in detections:
        if d.corner_count < 6:
            continue
        ids = np.asarray(d.ids, dtype=int)
        obj.append(all_obj[ids].reshape(-1, 1, 3))
        img.append(np.asarray(d.corners, dtype=np.float32).reshape(-1, 1, 2))
    if len(obj) < 10:
        raise ValueError(f"有效内参视角只有 {len(obj)} 个，至少需要 10 个，建议 30 个")
    if initial is not None:
        if (initial.width, initial.height) != (width, height):
            raise ValueError("初始内参与采集分辨率不一致")
        K = initial.K.copy()
        D = np.zeros((5, 1), dtype=float)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
    else:
        K = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
                     dtype=float)
        D = np.zeros((5, 1), dtype=float)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
    result = cv2.calibrateCameraExtended(obj, img, (width, height), K, D, flags=flags)
    rms, K, D, _rv, _tv, _si, _se, per_view = result
    intr = CameraIntrinsics(
        width=width, height=height, camera_matrix=K.tolist(),
        distortion=D.reshape(-1).tolist(), serial=serial,
        model="opencv_brown", source="charuco",
        rms_px=float(rms), views=len(obj), created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    ).validate()
    return intr, np.asarray(per_view).reshape(-1).astype(float).tolist()


@dataclass
class PosePair:
    """一个静止标定姿态的平均结果。"""

    index: int
    time_ns: int
    V_T_T: list[list[float]]
    C_T_K: list[list[float]]
    image: str = ""
    charuco_corners: int = 0
    reprojection_rms_px: float = 0.0
    burst_frames: int = 1
    tracker_translation_span_m: float = 0.0
    tracker_rotation_span_deg: float = 0.0
    board_translation_span_m: float = 0.0
    board_rotation_span_deg: float = 0.0

    def validate(self) -> "PosePair":
        _validate_transform(self.V_T_T, "V_T_T")
        _validate_transform(self.C_T_K, "C_T_K")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PosePair":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d}).validate()


def save_pose_pairs(dataset_dir: str | Path, pairs: Sequence[PosePair], metadata: dict) -> None:
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata)
    meta.update({"format": FORMAT_VERSION, "kind": "vive_d435i_handeye_dataset",
                 "samples": len(pairs)})
    (root / "capture.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    lines = "".join(json.dumps(p.to_dict(), ensure_ascii=False) + "\n" for p in pairs)
    (root / "samples.jsonl").write_text(lines, encoding="utf-8")


def load_pose_pairs(dataset_dir: str | Path) -> tuple[list[PosePair], dict]:
    root = Path(dataset_dir)
    meta = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    if meta.get("kind") != "vive_d435i_handeye_dataset" or meta.get("format") != FORMAT_VERSION:
        raise ValueError("不认识的手眼采集数据格式")
    pairs = [PosePair.from_dict(json.loads(line)) for line in
             (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(pairs) != meta.get("samples"):
        raise ValueError(f"capture.json 写 {meta.get('samples')} 个样本，实际 {len(pairs)} 个")
    return pairs, meta


def closure_transforms(pairs: Sequence[PosePair], T_T_C):
    np = _np()
    X = np.asarray(T_T_C, dtype=float)
    return np.stack([np.asarray(p.V_T_T) @ X @ np.asarray(p.C_T_K) for p in pairs])


def closure_statistics(pairs: Sequence[PosePair], T_T_C, indices: Sequence[int] | None = None):
    np = _np()
    chosen = list(pairs) if indices is None else [pairs[i] for i in indices]
    Ys = closure_transforms(chosen, T_T_C)
    Y = average_transforms(Ys)
    trans, rot = [], []
    for y in Ys:
        dt, dr = transform_distance(Y, y)
        trans.append(dt)
        rot.append(dr)
    trans_a, rot_a = np.asarray(trans), np.asarray(rot)
    def stats(v):
        return {"median": float(np.median(v)), "rms": float(np.sqrt(np.mean(v * v))),
                "p95": float(np.percentile(v, 95)), "max": float(v.max())}
    return Y, {"samples": len(chosen), "translation_m": stats(trans_a),
               "rotation_deg": stats(rot_a), "translation_errors_m": trans,
               "rotation_errors_deg": rot}


def motion_diagnostics(pairs: Sequence[PosePair]) -> dict:
    np = _np()
    Rotation = _rotation()
    if len(pairs) < 2:
        return {"max_translation_m": 0.0, "max_rotation_deg": 0.0,
                "rotation_excitation_singular_values": [0, 0, 0]}
    base = np.asarray(pairs[0].V_T_T)
    base_inv = invert_transform(base)
    rv, trans = [], []
    for p in pairs[1:]:
        d = base_inv @ np.asarray(p.V_T_T)
        rv.append(Rotation.from_matrix(d[:3, :3]).as_rotvec())
        trans.append(float(np.linalg.norm(d[:3, 3])))
    sv = np.linalg.svd(np.asarray(rv), compute_uv=False)
    return {"max_translation_m": max(trans),
            "max_rotation_deg": float(max(np.linalg.norm(v) for v in rv) * 180 / math.pi),
            "rotation_excitation_singular_values": sv.tolist()}


def _opencv_handeye(pairs: Sequence[PosePair], method_name: str):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError('需要 opencv-contrib-python：pip install -e ".[calibration]"') from exc
    np = _np()
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    if method_name not in methods:
        raise ValueError(f"未知 hand-eye 方法 {method_name}")
    A = [np.asarray(p.V_T_T, dtype=float) for p in pairs]
    B = [np.asarray(p.C_T_K, dtype=float) for p in pairs]
    R, t = cv2.calibrateHandEye(
        [x[:3, :3] for x in A], [x[:3, 3].reshape(3, 1) for x in A],
        [x[:3, :3] for x in B], [x[:3, 3].reshape(3, 1) for x in B],
        method=methods[method_name])
    X = make_transform(R, np.asarray(t).reshape(3))
    return _validate_transform(X, "T_T_C")


def _refine_xy(pairs: Sequence[PosePair], X0, indices: Sequence[int]):
    np = _np()
    Rotation = _rotation()
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover
        raise ImportError('联合优化需要 scipy：pip install -e ".[calibration]"') from exc
    selected = [pairs[i] for i in indices]
    Y0 = average_transforms(closure_transforms(selected, X0))

    def pack(X, Y):
        return np.r_[X[:3, 3], Rotation.from_matrix(X[:3, :3]).as_rotvec(),
                     Y[:3, 3], Rotation.from_matrix(Y[:3, :3]).as_rotvec()]

    def unpack(x):
        X = make_transform(Rotation.from_rotvec(x[3:6]).as_matrix(), x[:3])
        Y = make_transform(Rotation.from_rotvec(x[9:12]).as_matrix(), x[6:9])
        return X, Y

    sigma_t = 0.01
    sigma_r = math.radians(1.0)

    def residual(x):
        X, Y = unpack(x)
        out = []
        Yi = invert_transform(Y)
        for p in selected:
            pred = np.asarray(p.V_T_T) @ X @ np.asarray(p.C_T_K)
            d = Yi @ pred
            out.extend((d[:3, 3] / sigma_t).tolist())
            out.extend((Rotation.from_matrix(d[:3, :3]).as_rotvec() / sigma_r).tolist())
        return np.asarray(out)

    opt = least_squares(residual, pack(X0, Y0), loss="soft_l1", f_scale=1.0,
                        max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    X, Y = unpack(opt.x)
    return X, Y, {"success": bool(opt.success), "cost": float(opt.cost),
                  "nfev": int(opt.nfev), "message": str(opt.message)}


def _outlier_indices(stats: dict, indices: Sequence[int]) -> tuple[list[int], list[int]]:
    np = _np()
    te = np.asarray(stats["translation_errors_m"], dtype=float)
    re = np.asarray(stats["rotation_errors_deg"], dtype=float)
    # 两种误差归一后取较坏者。MAD 为零时仍保留一个工程下限，避免过度剔除。
    def limit(v, floor):
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        return max(floor, med + 3.5 * 1.4826 * mad)
    tl, rl = limit(te, 0.015), limit(re, 1.0)
    keep, drop = [], []
    for original, t, r in zip(indices, te, re):
        (keep if t <= tl and r <= rl else drop).append(original)
    return keep, drop


def solve_handeye(pairs: Sequence[PosePair], method: str = "auto", refine: bool = True) -> dict:
    """比较 OpenCV 多种解法，鲁棒联合优化 ``T_T_C`` 与 ``V_T_K``。"""
    if len(pairs) < 10:
        raise ValueError(f"手眼样本只有 {len(pairs)} 个，至少 10 个，建议 30–50 个")
    for p in pairs:
        p.validate()
    diag = motion_diagnostics(pairs)
    if diag["max_rotation_deg"] < 15:
        raise ValueError(f"最大旋转只有 {diag['max_rotation_deg']:.1f}°，手眼外参不可观；"
                         "至少绕两个方向各转 20–40°")
    sv = diag["rotation_excitation_singular_values"]
    if len(sv) >= 2 and sv[1] < math.radians(5):
        raise ValueError("旋转基本只绕一个轴，手眼外参退化；加入 roll/pitch/yaw 多轴姿态")

    names = [method] if method != "auto" else ["park", "horaud", "tsai", "andreff", "daniilidis"]
    holdout = [i for i in range(len(pairs)) if i % 5 == 0]
    train = [i for i in range(len(pairs)) if i not in holdout]
    candidate_rows = []
    for name in names:
        try:
            X = _opencv_handeye([pairs[i] for i in train], name)
            _, st = closure_statistics(pairs, X, holdout)
            score = st["translation_m"]["rms"] + math.radians(
                st["rotation_deg"]["rms"]) * 0.10
            candidate_rows.append({"method": name, "score": score,
                                   "holdout": st, "T_T_C": X})
        except Exception as exc:  # 不让单个 OpenCV 方法毁掉全部求解
            candidate_rows.append({"method": name, "error": str(exc)})
    valid = [c for c in candidate_rows if "T_T_C" in c and _np().isfinite(c["T_T_C"]).all()]
    if not valid:
        raise ValueError("所有 OpenCV hand-eye 方法都失败：" +
                         "; ".join(f"{c['method']}: {c.get('error')}" for c in candidate_rows))
    best = min(valid, key=lambda c: c["score"])
    chosen = best["method"]
    X = _opencv_handeye(pairs, chosen)  # 选完算法后，用全部样本重算
    inliers = list(range(len(pairs)))
    refine_info = {"enabled": bool(refine), "success": True, "nfev": 0}
    if refine:
        X, _Y, refine_info = _refine_xy(pairs, X, inliers)
    _, first_stats = closure_statistics(pairs, X, inliers)
    kept, dropped = _outlier_indices(first_stats, inliers)
    if len(kept) >= 10 and dropped:
        X = _opencv_handeye([pairs[i] for i in kept], chosen)
        if refine:
            X, _Y, refine_info = _refine_xy(pairs, X, kept)
        inliers = kept
    Y, inlier_stats = closure_statistics(pairs, X, inliers)
    _, all_stats = closure_statistics(pairs, X)
    _, holdout_stats = closure_statistics(pairs, X, holdout)

    def public_candidate(c):
        return {k: v for k, v in c.items() if k != "T_T_C"}
    pos, quat = transform_to_pose(X)
    y_pos, y_quat = transform_to_pose(Y)
    return {
        "format": FORMAT_VERSION, "kind": "vive_d435i_handeye",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "convention": {"V_T_T": "tracker body -> Z-up VIVE map",
                       "C_T_K": "ChArUco board -> D435i color optical",
                       "T_T_C": "D435i color optical -> tracker body",
                       "equation": "V_T_T @ T_T_C @ C_T_K = V_T_K"},
        "T_T_C": _matrix_list(X), "translation_m": pos, "quaternion_xyzw": quat,
        "calibration_session_V_T_K": _matrix_list(Y),
        "calibration_session_translation_m": y_pos,
        "calibration_session_quaternion_xyzw": y_quat,
        "solver": {"selected_method": chosen,
                   "candidates": [public_candidate(c) for c in candidate_rows],
                   "refinement": refine_info},
        "motion": diag,
        "samples": {"total": len(pairs), "inliers": inliers,
                    "outliers": sorted(set(range(len(pairs))) - set(inliers)),
                    "holdout": holdout},
        "closure": {"inliers": inlier_stats, "all": all_stats,
                    "holdout": holdout_stats},
    }


def save_handeye(path: str | Path, result: dict, dataset_meta: dict | None = None) -> None:
    out = dict(result)
    if dataset_meta is not None:
        out["capture"] = dataset_meta
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def load_handeye(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("kind") != "vive_d435i_handeye" or d.get("format") != FORMAT_VERSION:
        raise ValueError("不认识的 hand-eye 文件")
    _validate_transform(d["T_T_C"], "T_T_C")
    return d


def estimate_session_origin(pairs: Sequence[PosePair], T_T_C, O_T_K) -> dict:
    """从当前会话看板样本求 ``O_T_V``。"""
    np = _np()
    if len(pairs) < 5:
        raise ValueError("原点注册至少需要 5 个有效帧")
    X = _validate_transform(T_T_C, "T_T_C")
    O_T_K = _validate_transform(O_T_K, "O_T_K")
    Y, stats = closure_statistics(pairs, X)
    O_T_V = O_T_K @ invert_transform(Y)
    pos, quat = transform_to_pose(O_T_V)
    return {"format": FORMAT_VERSION, "kind": "vive_d435i_session_origin",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "convention": {"O_T_V": "current Z-up VIVE map -> odometry/world",
                           "equation": "O_T_V = O_T_K @ inverse(mean(V_T_T @ T_T_C @ C_T_K))"},
            "O_T_V": _matrix_list(O_T_V), "translation_m": pos,
            "quaternion_xyzw": quat, "mean_V_T_K": _matrix_list(Y),
            "quality": stats, "samples": len(pairs)}


def save_origin(path: str | Path, result: dict, handeye_path: str | Path = "") -> None:
    out = dict(result)
    if handeye_path:
        hp = Path(handeye_path)
        out["handeye_file"] = str(hp)
        out["handeye_sha256"] = hashlib.sha256(hp.read_bytes()).hexdigest()
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def load_origin(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("kind") != "vive_d435i_session_origin" or d.get("format") != FORMAT_VERSION:
        raise ValueError("不认识的会话原点文件")
    _validate_transform(d["O_T_V"], "O_T_V")
    return d


def quality_ok(stats: dict, translation_p95_m: float = 0.02,
               rotation_p95_deg: float = 1.0) -> tuple[bool, list[str]]:
    problems = []
    if stats["translation_m"]["p95"] > translation_p95_m:
        problems.append(f"位置闭环 p95 {stats['translation_m']['p95']*1000:.1f} mm > "
                        f"{translation_p95_m*1000:.1f} mm")
    if stats["rotation_deg"]["p95"] > rotation_p95_deg:
        problems.append(f"姿态闭环 p95 {stats['rotation_deg']['p95']:.2f}° > "
                        f"{rotation_p95_deg:.2f}°")
    return not problems, problems
