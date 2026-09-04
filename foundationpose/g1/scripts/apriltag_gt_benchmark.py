import os, sys, base64, time, csv, json, math
from pathlib import Path

import cv2, zmq, msgpack, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from estimater import *

ENDPOINT = 'tcp://192.168.123.164:5555'
INIT = ROOT / 'g1/data/live_init'
CFG_PATH = ROOT / 'g1/config/apriltag_gt_placeholder.json'
OUT = ROOT / 'g1/results/apriltag_gt_benchmark'
K = np.loadtxt(INIT / 'cam_K.txt').reshape(3, 3)

TRACK_ITERS = 1
REGISTER_ITERS = 5
VIS_EVERY = 3


def decode(payload):
    data = msgpack.unpackb(payload, raw=False)
    out = {}
    for k, v in data['images'].items():
        b = base64.b64decode(v) if isinstance(v, str) else v
        out[k] = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    return out


def recv_rgbd(sock):
    while True:
        im = decode(sock.recv())
        if 'ego_view' in im and 'ego_view_depth' in im:
            return im['ego_view'], im['ego_view_depth']


def depth_m(depth_u16):
    d = depth_u16.astype(np.float32) * 0.001
    d[(d < 0.001) | (d > 10.0)] = 0
    return d


def mask_ui(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    shown = bgr.copy()
    pts = []
    win = 'Initial FoundationPose mask - draw ONCE'

    def mouse(event, x, y, flags, param):
        nonlocal shown
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            shown = bgr.copy()
            if len(pts) > 1:
                cv2.polylines(shown, [np.asarray(pts)], False, (0, 255, 0), 2)
            for p in pts:
                cv2.circle(shown, p, 4, (0, 0, 255), -1)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse)
    print('Draw box mask once: left-click corners | ENTER accept | R reset | ESC cancel')
    while True:
        frame = shown.copy()
        cv2.putText(frame, 'ENTER accept | R reset | ESC cancel',
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(pts) >= 3:
            mask = np.zeros(rgb.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.asarray(pts)], 1)
            cv2.destroyWindow(win)
            return mask.astype(bool)
        if key == ord('r'):
            pts.clear()
            shown = bgr.copy()
        if key == 27:
            cv2.destroyWindow(win)
            return None


def rotation_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = (np.trace(R) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def append_csv(path, row):
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def make_detector(dictionary_name):
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('cv2.aruco is unavailable. Install/use an OpenCV build with aruco support.')
    aruco = cv2.aruco
    if not hasattr(aruco, dictionary_name):
        raise RuntimeError(f'Unknown AprilTag dictionary: {dictionary_name}')
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    if hasattr(aruco, 'ArucoDetector'):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)
    params = aruco.DetectorParameters_create()
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def estimate_tag_pose(rgb, detect_fn, wanted_id, tag_size_m):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, rejected = detect_fn(gray)
    if ids is None:
        return None, None
    ids_flat = ids.reshape(-1)
    matches = np.where(ids_flat == wanted_id)[0]
    if len(matches) == 0:
        return None, None

    c = np.asarray(corners[int(matches[0])], dtype=np.float32).reshape(4, 2)
    h = tag_size_m * 0.5
    obj = np.asarray([
        [-h,  h, 0.0],
        [ h,  h, 0.0],
        [ h, -h, 0.0],
        [-h, -h, 0.0],
    ], dtype=np.float32)

    flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=flag)
    if not ok:
        return None, c
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T, c


def draw_tag_axes(bgr, T_cam_tag, tag_size_m):
    if T_cam_tag is None:
        return bgr
    rvec, _ = cv2.Rodrigues(T_cam_tag[:3, :3])
    tvec = T_cam_tag[:3, 3].reshape(3, 1)
    cv2.drawFrameAxes(bgr, K, None, rvec, tvec, tag_size_m * 0.7, 2)
    return bgr


def main():
    set_logging_format()
    set_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(CFG_PATH.read_text())
    wanted_id = int(cfg['marker_id'])
    tag_size_m = float(cfg['marker_size_m'])
    T_tag_box = np.asarray(cfg['T_tag_box'], dtype=np.float64).reshape(4, 4)
    detect_fn = make_detector(cfg['dictionary'])

    print('APRILTAG REFERENCE BENCHMARK')
    print('Marker family/id/size and measured in-plane offset are configured; validate the current box-face orientation assumption once from the overlay.')
    print(f"dictionary={cfg['dictionary']} | marker_id={wanted_id} | marker_size={tag_size_m:.3f} m")
    print('T_camera_box(tag reference) = T_camera_tag @ T_tag_box')
    print('Draw FoundationPose mask once, then move the box freely. Q quits.')

    mesh = trimesh.load(ROOT / 'box.obj')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    center_tf = np.linalg.inv(to_origin)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
        debug_dir=str(OUT), debug=0, glctx=dr.RasterizeCudaContext())

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, '')
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(ENDPOINT)

    csv_path = OUT / 'apriltag_vs_foundationpose.csv'

    try:
        rgb0, depth_raw0 = recv_rgbd(sock)
        depth0 = depth_m(depth_raw0)
        mask = mask_ui(rgb0)
        if mask is None:
            print('Cancelled.')
            return

        cv2.imwrite(str(OUT / 'initial_rgb.png'), cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT / 'initial_depth.png'), depth_raw0)
        cv2.imwrite(str(OUT / 'initial_mask.png'), mask.astype(np.uint8) * 255)

        t0 = time.perf_counter()
        fp_pose = est.register(K=K, rgb=rgb0, depth=depth0, ob_mask=mask,
                               iteration=REGISTER_ITERS)
        register_ms = (time.perf_counter() - t0) * 1000.0
        if not np.isfinite(fp_pose).all():
            raise RuntimeError('FoundationPose initial registration returned non-finite pose')
        np.savetxt(OUT / 'initial_foundationpose.txt', fp_pose)
        print(f'FoundationPose initial registration OK: {register_ms:.1f} ms')

        frame_id = 0
        detected_count = 0
        while True:
            rgb, depth_raw = recv_rgbd(sock)
            depth = depth_m(depth_raw)

            t1 = time.perf_counter()
            fp_pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=TRACK_ITERS)
            fp_ms = (time.perf_counter() - t1) * 1000.0
            frame_id += 1

            T_cam_tag, tag_corners = estimate_tag_pose(
                rgb, detect_fn, wanted_id, tag_size_m)

            fp_center = fp_pose @ center_tf
            fp_distance = float(np.linalg.norm(fp_center[:3, 3]))
            fp_z = float(fp_center[2, 3])

            tag_detected = T_cam_tag is not None
            tag_distance = float('nan')
            tag_z = float('nan')
            translation_error_mm = float('nan')
            rotation_error_deg = float('nan')
            distance_error_mm = float('nan')
            T_cam_box_tag = None

            if tag_detected:
                detected_count += 1
                T_cam_box_tag = T_cam_tag @ T_tag_box
                tag_distance = float(np.linalg.norm(T_cam_box_tag[:3, 3]))
                tag_z = float(T_cam_box_tag[2, 3])
                translation_error_mm = float(
                    np.linalg.norm(fp_pose[:3, 3] - T_cam_box_tag[:3, 3]) * 1000.0)
                rotation_error_deg = float(
                    rotation_angle_deg(fp_pose[:3, :3], T_cam_box_tag[:3, :3]))
                distance_error_mm = float(abs(fp_distance - tag_distance) * 1000.0)

            row = {
                'frame': frame_id,
                'timestamp': time.time(),
                'tag_detected': tag_detected,
                'foundationpose_track_ms': fp_ms,
                'foundationpose_distance_m': fp_distance,
                'foundationpose_z_m': fp_z,
                'apriltag_distance_m': tag_distance,
                'apriltag_z_m': tag_z,
                'distance_error_mm': distance_error_mm,
                'translation_error_mm': translation_error_mm,
                'rotation_error_deg': rotation_error_deg,
            }
            append_csv(csv_path, row)

            if frame_id % VIS_EVERY == 0:
                vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=fp_center, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=fp_center, scale=0.1, K=K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                if tag_corners is not None:
                    poly = np.round(tag_corners).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(bgr, [poly], True, (255, 255, 0), 2)
                bgr = draw_tag_axes(bgr, T_cam_tag, tag_size_m)

                if tag_detected:
                    text = (f'FP {fp_distance:.2f}m | TAG {tag_distance:.2f}m | '
                            f'd={translation_error_mm:.0f}mm r={rotation_error_deg:.1f}deg')
                else:
                    text = f'FP {fp_distance:.2f}m | TAG id {wanted_id} NOT DETECTED'
                cv2.putText(bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (0, 255, 0), 1)
                cv2.imshow('AprilTag vs FoundationPose', bgr)

            if frame_id % 30 == 0:
                print(f'frames={frame_id} | tag detections={detected_count} | '
                      f'FP distance={fp_distance:.3f} m | '
                      + (f'tag distance={tag_distance:.3f} m | error={translation_error_mm:.1f} mm / {rotation_error_deg:.2f} deg'
                         if tag_detected else 'tag not detected'))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        sock.close(0)
        ctx.term()
        print('Saved:', csv_path)
        print('AprilTag is a reference estimate. Validate the configured box-face/orientation convention from the overlay before treating full pose error as ground truth.')


if __name__ == '__main__':
    main()
