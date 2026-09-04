#!/usr/bin/env python3

import base64
import csv
import sys
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import pyrealsense2 as rs
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent))   # g1_config 在任意 cwd 下可导入
import g1_config as cfg


# ============================================================
# CONFIG
# ============================================================

G1_ENDPOINT = cfg.G1_ENDPOINT

# G1 640x480 RGB intrinsics (see g1_config.py)
K_G1 = np.asarray(cfg.K_G1, dtype=np.float64)

# G1 pipeline did not provide/store distortion.
D_G1 = np.asarray(cfg.D_G1, dtype=np.float64).reshape(5, 1)

# ChArUco (see g1_config.py)
SQUARES_X = cfg.SQUARES_X
SQUARES_Y = cfg.SQUARES_Y
SQUARE_LENGTH = cfg.SQUARE_LENGTH_M
MARKER_LENGTH = cfg.MARKER_LENGTH_M

MIN_CHARUCO_CORNERS = cfg.MIN_CORNERS

# Hold board still during these
CALIBRATION_SAMPLES = 30
VALIDATION_SAMPLES = 10

AXIS_LENGTH = 0.08  # 8 cm

OUT = cfg.DUAL_CAM_RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHARUCO
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    getattr(cv2.aruco, cfg.BOARD_DICT)
)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)

detector = cv2.aruco.CharucoDetector(board)

try:
    BOARD_CORNERS_3D = np.asarray(
        board.getChessboardCorners(),
        dtype=np.float32
    ).reshape(-1, 3)
except Exception:
    BOARD_CORNERS_3D = None


# ============================================================
# TRANSFORM UTILS
# ============================================================

def rt_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec).reshape(3)
    return T


def T_to_rt(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)
    return rvec, tvec


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]

    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def average_transforms(Ts):
    """Average translation + project mean rotation back onto SO(3)."""
    t = np.mean([T[:3, 3] for T in Ts], axis=0)

    M = np.sum([T[:3, :3] for T in Ts], axis=0)
    U, _, Vt = np.linalg.svd(M)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pose_error(T_measured, T_predicted):
    """
    Error between two poses of the same object
    expressed in the same camera frame.
    """
    D = inv_T(T_measured) @ T_predicted

    trans_mm = np.linalg.norm(D[:3, 3]) * 1000.0

    trace = np.trace(D[:3, :3])
    c = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = np.degrees(np.arccos(c))

    return trans_mm, rot_deg


def rotation_spread(T_ref, T):
    return pose_error(T_ref, T)


# ============================================================
# CHARUCO POSE
# ============================================================

def detect_charuco(image_bgr, K, D):
    vis = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        detector.detectBoard(gray)

    # OpenCV 5 Python shape normalization
    if marker_ids is not None:
        marker_ids = np.asarray(
            marker_ids,
            dtype=np.int32
        ).reshape(-1, 1)

    if charuco_ids is not None:
        charuco_ids = np.asarray(
            charuco_ids,
            dtype=np.int32
        ).reshape(-1, 1)

    if charuco_corners is not None:
        charuco_corners = np.asarray(
            charuco_corners,
            dtype=np.float32
        ).reshape(-1, 1, 2)

    # Safety check
    if charuco_corners is not None and charuco_ids is not None:
        if len(charuco_corners) != len(charuco_ids):
            print(
                "WARNING: ChArUco mismatch:",
                "corners =", len(charuco_corners),
                "ids =", len(charuco_ids)
            )
            charuco_corners = None
            charuco_ids = None

    # Draw detected ArUco markers
    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(
            vis,
            marker_corners,
            marker_ids
        )

    count = 0 if charuco_ids is None else len(charuco_ids)

    # Draw interpolated ChArUco corners
    if (
        charuco_corners is not None
        and charuco_ids is not None
        and count > 0
    ):
        cv2.aruco.drawDetectedCornersCharuco(
            vis,
            charuco_corners,
            charuco_ids,
            (255, 255, 0)
        )

    result = {
        "T": None,
        "corners": charuco_corners,
        "ids": charuco_ids,
        "count": count,
        "rmse": None,
    }

    if (
        charuco_corners is None
        or charuco_ids is None
        or count < MIN_CHARUCO_CORNERS
    ):
        return vis, result

    try:
        obj_pts, img_pts = board.matchImagePoints(
            charuco_corners,
            charuco_ids
        )

        obj_pts = np.asarray(
            obj_pts,
            dtype=np.float32
        ).reshape(-1, 3)

        img_pts = np.asarray(
            img_pts,
            dtype=np.float32
        ).reshape(-1, 2)

        if len(obj_pts) < 4:
            return vis, result

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok:
            return vis, result

        T = rt_to_T(rvec, tvec)

        cv2.drawFrameAxes(
            vis,
            K,
            D,
            rvec,
            tvec,
            AXIS_LENGTH,
            3
        )

        projected, _ = cv2.projectPoints(
            obj_pts,
            rvec,
            tvec,
            K,
            D
        )

        projected = projected.reshape(-1, 2)

        rmse = np.sqrt(
            np.mean(
                np.sum(
                    (projected - img_pts) ** 2,
                    axis=1
                )
            )
        )

        result["T"] = T
        result["rmse"] = float(rmse)

    except Exception as e:
        print("Pose estimation error:", repr(e))

    return vis, result


# ============================================================
# VISUAL VALIDATION
# ============================================================

def draw_prediction_on_external(
    vis,
    ext_result,
    T_ext_board_predicted,
    K_ext,
    D_ext
):
    """
    Draw predicted ChArUco corner positions in MAGENTA.

    Cyan = directly measured by external camera.
    Magenta = predicted from G1 camera + calibrated transform.
    Lines connect measurement to prediction.
    """

    if (
        BOARD_CORNERS_3D is None
        or ext_result["ids"] is None
        or ext_result["corners"] is None
    ):
        return

    ids = np.asarray(ext_result["ids"]).reshape(-1).astype(int)
    measured = np.asarray(
        ext_result["corners"]
    ).reshape(-1, 2)

    valid = (ids >= 0) & (ids < len(BOARD_CORNERS_3D))

    if not np.any(valid):
        return

    ids = ids[valid]
    measured = measured[valid]

    obj = BOARD_CORNERS_3D[ids]

    rvec, tvec = T_to_rt(T_ext_board_predicted)

    pred, _ = cv2.projectPoints(
        obj,
        rvec,
        tvec,
        K_ext,
        D_ext
    )

    pred = pred.reshape(-1, 2)

    for m, p in zip(measured, pred):
        mx, my = np.round(m).astype(int)
        px, py = np.round(p).astype(int)

        # predicted point
        cv2.circle(
            vis,
            (px, py),
            6,
            (255, 0, 255),
            2
        )

        # error line
        cv2.line(
            vis,
            (mx, my),
            (px, py),
            (255, 0, 255),
            1
        )


# ============================================================
# TEXT
# ============================================================

def put_text(img, text, y, color=(0, 255, 0), scale=0.55):
    cv2.putText(
        img,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# SAVE
# ============================================================

def save_calibration(T_ext_g1, K_ext, D_ext):
    np.savetxt(
        OUT / "T_external_from_g1_camera.txt",
        T_ext_g1,
        fmt="%.10f"
    )

    np.savetxt(
        OUT / "T_g1_camera_from_external.txt",
        inv_T(T_ext_g1),
        fmt="%.10f"
    )

    np.savez(
        OUT / "dual_camera_calibration.npz",
        T_external_from_g1=T_ext_g1,
        T_g1_from_external=inv_T(T_ext_g1),
        K_g1=K_G1,
        D_g1=D_G1,
        K_external=K_ext,
        D_external=D_ext,
    )

    with open(OUT / "README.txt", "w") as f:
        f.write(
            "T_external_from_g1_camera maps a point from the\n"
            "G1 RGB optical-camera frame into the external\n"
            "D435i color optical-camera frame.\n\n"
            "p_external = T_external_from_g1_camera @ p_g1\n\n"
            "Board:\n"
            f"{SQUARES_X} x {SQUARES_Y} squares\n"
            f"{cfg.BOARD_DICT}\n"
            f"square = {SQUARE_LENGTH:.3f} m\n"
            f"marker = {MARKER_LENGTH:.3f} m\n"
        )


def append_validation(run_id, index, trans_mean, trans_std,
                      rot_mean, rot_std):
    """Append one validation set to validation_runs.csv.

    run_id is the timestamp of the calibration this set belongs to
    (printed at CALIBRATION COMPLETE). validation_id restarts at 1
    for every calibration, so rows are only unambiguous together
    with run_id. The legacy validation.csv (no run_id) is left alone.
    """
    path = OUT / "validation_runs.csv"
    new_file = not path.exists()

    with open(path, "a", newline="") as f:
        w = csv.writer(f)

        if new_file:
            w.writerow([
                "run_id",
                "validation_id",
                "timestamp",
                "translation_mean_mm",
                "translation_std_mm",
                "rotation_mean_deg",
                "rotation_std_deg",
            ])

        w.writerow([
            run_id,
            index,
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            trans_mean,
            trans_std,
            rot_mean,
            rot_std,
        ])


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # G1 ZMQ
    # --------------------------------------------------------
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)

    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 3000)

    sock.connect(G1_ENDPOINT)

    # --------------------------------------------------------
    # External D435i
    # --------------------------------------------------------
    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(
        rs.stream.color,
        640,
        480,
        rs.format.bgr8,
        30
    )

    profile = pipe.start(cfg)

    color_profile = (
        profile
        .get_stream(rs.stream.color)
        .as_video_stream_profile()
    )

    intr = color_profile.get_intrinsics()

    K_EXT = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    D_EXT = np.asarray(
        intr.coeffs,
        dtype=np.float64
    ).reshape(-1, 1)

    print()
    print("==============================================")
    print("DUAL CAMERA CHARUCO CALIBRATION")
    print("==============================================")
    print()
    print(f"Board: {SQUARES_X}x{SQUARES_Y} / {cfg.BOARD_DICT}")
    print(f"Square: {SQUARE_LENGTH * 1000:.0f} mm")
    print(f"Marker: {MARKER_LENGTH * 1000:.0f} mm")
    print()
    print("G1 endpoint:", G1_ENDPOINT)
    print()
    print("G1 K:")
    print(K_G1)
    print()
    print("External K:")
    print(K_EXT)
    print()
    print("External distortion:", D_EXT.reshape(-1))
    print()
    print("Controls:")
    print("  C = calibrate (hold board still)")
    print("  V = validation sample (move board, hold still)")
    print("  R = reset")
    print("  Q = quit")
    print()
    print("IMPORTANT:")
    print("Do NOT move either camera after calibration.")
    print("Hold the ChArUco board still during C and V.")
    print("==============================================")
    print()

    T_EXT_G1 = None

    calibration_active = False
    calibration_samples = []

    validation_active = False
    validation_errors = []
    validation_id = 0
    run_id = None          # set when a calibration completes

    last_current_error = None

    cv2.namedWindow(
        "Dual Camera ChArUco Calibration",
        cv2.WINDOW_NORMAL
    )

    try:
        while True:

            # ------------------------------------------------
            # Receive G1
            # ------------------------------------------------
            try:
                payload = sock.recv()
            except zmq.Again:
                print("Waiting for G1 camera...")
                continue

            data = msgpack.unpackb(payload, raw=False)

            if "ego_view" not in data["images"]:
                continue

            v = data["images"]["ego_view"]

            buf = (
                base64.b64decode(v)
                if isinstance(v, str)
                else v
            )

            g1 = cv2.imdecode(
                np.frombuffer(buf, np.uint8),
                cv2.IMREAD_COLOR
            )

            # G1 server sends RGB
            g1 = cv2.cvtColor(
                g1,
                cv2.COLOR_RGB2BGR
            )

            # ------------------------------------------------
            # Receive external D435i
            # ------------------------------------------------
            frames = pipe.wait_for_frames(3000)
            color = frames.get_color_frame()

            if not color:
                continue

            ext = np.asanyarray(
                color.get_data()
            ).copy()

            # ------------------------------------------------
            # Detect ChArUco
            # ------------------------------------------------
            g1_vis, g1_res = detect_charuco(
                g1,
                K_G1,
                D_G1
            )

            ext_vis, ext_res = detect_charuco(
                ext,
                K_EXT,
                D_EXT
            )

            T_G1_BOARD = g1_res["T"]
            T_EXT_BOARD = ext_res["T"]

            # ------------------------------------------------
            # Camera labels
            # ------------------------------------------------
            put_text(
                g1_vis,
                "G1 CAMERA",
                28,
                (0, 255, 0),
                0.75
            )

            put_text(
                ext_vis,
                "EXTERNAL D435i",
                28,
                (0, 255, 0),
                0.75
            )

            put_text(
                g1_vis,
                f"ChArUco corners: {g1_res['count']}",
                55
            )

            put_text(
                ext_vis,
                f"ChArUco corners: {ext_res['count']}",
                55
            )

            if g1_res["rmse"] is not None:
                put_text(
                    g1_vis,
                    f"PnP reproj: {g1_res['rmse']:.2f} px",
                    80
                )

            if ext_res["rmse"] is not None:
                put_text(
                    ext_vis,
                    f"PnP reproj: {ext_res['rmse']:.2f} px",
                    80
                )

            both_valid = (
                T_G1_BOARD is not None
                and T_EXT_BOARD is not None
            )

            # ------------------------------------------------
            # CALIBRATION COLLECTION
            # ------------------------------------------------
            if calibration_active:

                put_text(
                    g1_vis,
                    f"CALIBRATING {len(calibration_samples)}/{CALIBRATION_SAMPLES}",
                    110,
                    (0, 255, 255),
                    0.65
                )

                if both_valid:

                    # Board -> G1
                    # Board -> EXT
                    #
                    # EXT <- G1:
                    #
                    # T_EXT_G1 =
                    # T_EXT_BOARD * inv(T_G1_BOARD)

                    sample = (
                        T_EXT_BOARD
                        @ inv_T(T_G1_BOARD)
                    )

                    calibration_samples.append(sample)

                    if len(calibration_samples) >= CALIBRATION_SAMPLES:

                        T_EXT_G1 = average_transforms(
                            calibration_samples
                        )

                        run_id = time.strftime("%Y%m%d-%H%M%S")

                        save_calibration(
                            T_EXT_G1,
                            K_EXT,
                            D_EXT
                        )

                        spreads = [
                            rotation_spread(
                                T_EXT_G1,
                                T
                            )
                            for T in calibration_samples
                        ]

                        tr = np.array(
                            [x[0] for x in spreads]
                        )

                        rr = np.array(
                            [x[1] for x in spreads]
                        )

                        print()
                        print("==============================================")
                        print("CALIBRATION COMPLETE")
                        print("==============================================")
                        print()
                        print("T_external_from_g1_camera:")
                        print(T_EXT_G1)
                        print()
                        print(
                            f"Calibration-frame spread: "
                            f"{tr.mean():.2f} +/- {tr.std():.2f} mm"
                        )
                        print(
                            f"Rotation spread: "
                            f"{rr.mean():.3f} +/- {rr.std():.3f} deg"
                        )
                        print()
                        print("Saved to:")
                        print(OUT)
                        print("run_id:", run_id)
                        print()
                        print("Now MOVE the board somewhere else.")
                        print("Hold it still and press V.")
                        print("==============================================")
                        print()

                        calibration_active = False
                        calibration_samples = []

            # ------------------------------------------------
            # LIVE VALIDATION
            # ------------------------------------------------
            current_error = None

            if T_EXT_G1 is not None and both_valid:

                # Predict board pose in external camera
                # using ONLY G1 observation + calibration.
                T_EXT_BOARD_PRED = (
                    T_EXT_G1
                    @ T_G1_BOARD
                )

                trans_mm, rot_deg = pose_error(
                    T_EXT_BOARD,
                    T_EXT_BOARD_PRED
                )

                current_error = (
                    trans_mm,
                    rot_deg
                )

                last_current_error = current_error

                # Predicted points vs measured points
                draw_prediction_on_external(
                    ext_vis,
                    ext_res,
                    T_EXT_BOARD_PRED,
                    K_EXT,
                    D_EXT
                )

                put_text(
                    ext_vis,
                    f"LIVE ERROR: {trans_mm:.1f} mm / {rot_deg:.2f} deg",
                    110,
                    (255, 0, 255),
                    0.62
                )

                put_text(
                    ext_vis,
                    "Magenta = predicted from G1",
                    135,
                    (255, 0, 255),
                    0.5
                )

            # ------------------------------------------------
            # VALIDATION COLLECTION
            # ------------------------------------------------
            if validation_active:

                put_text(
                    ext_vis,
                    f"VALIDATING {len(validation_errors)}/{VALIDATION_SAMPLES}",
                    165,
                    (0, 255, 255),
                    0.6
                )

                if current_error is not None:

                    validation_errors.append(
                        current_error
                    )

                    if len(validation_errors) >= VALIDATION_SAMPLES:

                        arr = np.asarray(
                            validation_errors,
                            dtype=float
                        )

                        tm = arr[:, 0].mean()
                        ts = arr[:, 0].std()

                        rm = arr[:, 1].mean()
                        rsd = arr[:, 1].std()

                        validation_id += 1

                        append_validation(
                            run_id,
                            validation_id,
                            tm,
                            ts,
                            rm,
                            rsd
                        )

                        print()
                        print(
                            f"VALIDATION run {run_id} #{validation_id}: "
                            f"{tm:.2f} +/- {ts:.2f} mm | "
                            f"{rm:.3f} +/- {rsd:.3f} deg"
                        )

                        validation_active = False
                        validation_errors = []

            # ------------------------------------------------
            # Bottom information panel
            # ------------------------------------------------
            side = np.hstack([
                g1_vis,
                ext_vis
            ])

            panel = np.zeros(
                (125, side.shape[1], 3),
                dtype=np.uint8
            )

            if T_EXT_G1 is None:
                status = "NOT CALIBRATED - Hold board still in BOTH cameras and press C"
                color = (0, 200, 255)
            else:
                status = "CALIBRATED - Move board to a NEW pose, hold still, press V"
                color = (0, 255, 0)

            put_text(
                panel,
                status,
                28,
                color,
                0.62
            )

            put_text(
                panel,
                "C: calibrate    V: validate    R: reset    Q: quit",
                58,
                (220, 220, 220),
                0.55
            )

            if last_current_error is not None:
                put_text(
                    panel,
                    (
                        f"Current held-out consistency: "
                        f"{last_current_error[0]:.1f} mm   "
                        f"{last_current_error[1]:.2f} deg   "
                        f"| validation sets: {validation_id}"
                    ),
                    90,
                    (255, 0, 255),
                    0.55
                )

            view = np.vstack([
                side,
                panel
            ])

            cv2.imshow(
                "Dual Camera ChArUco Calibration",
                view
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            elif key in (ord("c"), ord("C")):

                if not both_valid:
                    print(
                        "Cannot calibrate: "
                        "ChArUco must be detected by BOTH cameras."
                    )
                    continue

                print()
                print(
                    "CALIBRATION STARTED - "
                    "KEEP BOARD COMPLETELY STILL"
                )

                T_EXT_G1 = None
                calibration_samples = []
                calibration_active = True

                validation_active = False
                validation_errors = []
                validation_id = 0
                last_current_error = None

            elif key in (ord("v"), ord("V")):

                if T_EXT_G1 is None:
                    print(
                        "Calibrate first with C."
                    )
                    continue

                if not both_valid:
                    print(
                        "Cannot validate: "
                        "board must be visible in BOTH cameras."
                    )
                    continue

                print()
                print(
                    "VALIDATION STARTED - "
                    "KEEP BOARD COMPLETELY STILL"
                )

                validation_errors = []
                validation_active = True

            elif key in (ord("r"), ord("R")):

                print("Calibration reset.")

                T_EXT_G1 = None

                calibration_active = False
                calibration_samples = []

                validation_active = False
                validation_errors = []

                validation_id = 0
                last_current_error = None

    finally:

        pipe.stop()

        sock.close(0)
        ctx.term()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
