import time, csv, json, base64, argparse, shutil
from pathlib import Path

import cv2
import zmq
import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINT = 'tcp://192.168.123.164:5555'
DEFAULT_K = ROOT / 'g1/data/live_init/cam_K.txt'
APRIL_CFG = ROOT / 'g1/config/apriltag_gt_placeholder.json'


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


def make_detector(dictionary_name):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    if hasattr(aruco, 'ArucoDetector'):
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        return lambda gray: detector.detectMarkers(gray)
    params = aruco.DetectorParameters_create()
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def estimate_tag_pose(rgb, detect_fn, wanted_id, tag_size_m, K):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detect_fn(gray)
    if ids is None:
        return None, None
    ids = ids.reshape(-1)
    idx = np.where(ids == wanted_id)[0]
    if len(idx) == 0:
        return None, None

    c = np.asarray(corners[int(idx[0])], dtype=np.float32).reshape(4, 2)
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


def box_corners(dims):
    hx, hy, hz = np.asarray(dims, dtype=float) * 0.5
    return np.asarray([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=np.float64)


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def project_box(points_box, T_cam_box, K):
    pts = (T_cam_box[:3, :3] @ points_box.T).T + T_cam_box[:3, 3]
    if np.any(pts[:, 2] <= 1e-6):
        return None
    uv = np.empty((len(pts), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2]
    uv[:, 1] = K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]
    return uv


def draw_box(bgr, uv):
    if uv is None:
        return
    p = np.round(uv).astype(np.int32)
    for a, b in EDGES:
        cv2.line(bgr, tuple(p[a]), tuple(p[b]), (0, 255, 255), 2, cv2.LINE_AA)


def depth_preview(depth):
    d = depth.astype(np.float32)
    d[d <= 0] = 0
    vis = np.clip(d / 3000.0 * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    cv2.putText(vis, 'DEPTH (0-3m)', (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    return vis


def parse_args():
    p = argparse.ArgumentParser(description='Record a rosbag-like offline RGB-D bundle from the G1 camera stream.')
    p.add_argument('--endpoint', default=DEFAULT_ENDPOINT)
    p.add_argument('--seconds', type=float, default=60.0,
                   help='Recording duration. Use <=0 to record until Q/Ctrl+C.')
    p.add_argument('--out', default='',
                   help='Output folder. Default: g1/data/offline_bundle_<timestamp>')
    p.add_argument('--every', type=int, default=1,
                   help='Save every Nth received RGB-D frame (default 1).')
    p.add_argument('--robot-marker-tf', default='',
                   help='Optional 4x4 txt file for one independently measured T_robot_marker reference pose.')
    p.add_argument('--no-preview', action='store_true',
                   help='Disable live RGB/depth visualization.')
    return p.parse_args()


def main():
    args = parse_args()
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = Path(args.out) if args.out else ROOT / f'g1/data/offline_bundle_{stamp}'
    rgb_dir = out / 'rgb'
    depth_dir = out / 'depth'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    K = np.loadtxt(DEFAULT_K).reshape(3, 3)
    np.savetxt(out / 'cam_K.txt', K)

    april_cfg = None
    detect_fn = None
    tag_id = None
    tag_size_m = None
    T_tag_box = None
    corners_box = None
    if APRIL_CFG.exists():
        shutil.copy2(APRIL_CFG, out / 'apriltag_config.json')
        april_cfg = json.loads(APRIL_CFG.read_text())
        if hasattr(cv2, 'aruco'):
            detect_fn = make_detector(april_cfg['dictionary'])
            tag_id = int(april_cfg['marker_id'])
            tag_size_m = float(april_cfg['marker_size_m'])
            if 'T_tag_box' in april_cfg and 'box_dimensions_m' in april_cfg:
                T_tag_box = np.asarray(april_cfg['T_tag_box'], dtype=np.float64).reshape(4, 4)
                corners_box = box_corners(april_cfg['box_dimensions_m'])

    if (ROOT / 'box.obj').exists():
        shutil.copy2(ROOT / 'box.obj', out / 'box.obj')

    if args.robot_marker_tf:
        T = np.loadtxt(Path(args.robot_marker_tf)).reshape(4, 4)
        np.savetxt(out / 'T_robot_marker_reference.txt', T)

    metadata = {
        'format': 'g1_offline_rgbd_bundle_v1',
        'created_unix_s': time.time(),
        'endpoint': args.endpoint,
        'rgb_key': 'ego_view',
        'depth_key': 'ego_view_depth',
        'depth_unit': 'uint16_mm',
        'camera_intrinsics_file': 'cam_K.txt',
        'apriltag_config_file': 'apriltag_config.json' if APRIL_CFG.exists() else None,
        'box_mesh_file': 'box.obj' if (ROOT / 'box.obj').exists() else None,
        'robot_stationary_during_capture': True,
        'T_robot_camera': None,
        'T_robot_marker_reference_file': 'T_robot_marker_reference.txt' if args.robot_marker_tf else None,
        'note': 'Raw RGB-D is sufficient to rerun FoundationPose and AprilTag offline. Robot is stationary. Add one verified static robot-camera transform later, or derive it from one independent robot-marker reference.'
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, '')
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(args.endpoint)

    f = (out / 'timestamps.csv').open('w', newline='')
    writer = csv.DictWriter(f, fieldnames=['frame', 'unix_s', 'relative_s', 'rgb_file', 'depth_file'])
    writer.writeheader()

    print('OFFLINE RGB-D BUNDLE RECORDER')
    print('output:', out)
    print('duration:', 'until Q/Ctrl+C' if args.seconds <= 0 else f'{args.seconds:.1f} s')
    print('preview: RGB + depth; AprilTag overlay enabled when detected')
    print('Q = stop recording')

    t0 = time.perf_counter()
    received = 0
    saved = 0
    try:
        while True:
            rgb, depth = recv_rgbd(sock)
            received += 1
            rel = time.perf_counter() - t0

            # Live preview runs on every received frame, independent of save decimation.
            if not args.no_preview:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                tag_detected = False
                if detect_fn is not None:
                    T_cam_tag, tag_corners = estimate_tag_pose(rgb, detect_fn, tag_id, tag_size_m, K)
                    if tag_corners is not None:
                        poly = np.round(tag_corners).astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(bgr, [poly], True, (255, 255, 0), 2)
                    if T_cam_tag is not None:
                        tag_detected = True
                        rvec, _ = cv2.Rodrigues(T_cam_tag[:3, :3])
                        cv2.drawFrameAxes(bgr, K, None, rvec, T_cam_tag[:3, 3], tag_size_m * 0.7, 2)
                        if T_tag_box is not None and corners_box is not None:
                            T_cam_box = T_cam_tag @ T_tag_box
                            draw_box(bgr, project_box(corners_box, T_cam_box, K))

                status = f'REC {rel:6.1f}s  saved={saved}  TAG={"YES" if tag_detected else "NO"}'
                cv2.putText(bgr, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)
                cv2.putText(bgr, 'Q stop | cyan=tag | yellow=box inferred from tag',
                            (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

                dvis = depth_preview(depth)
                if dvis.shape[:2] != bgr.shape[:2]:
                    dvis = cv2.resize(dvis, (bgr.shape[1], bgr.shape[0]))
                preview = np.hstack([bgr, dvis])
                cv2.imshow('G1 offline recorder - RGB / depth', preview)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    print('\nStopped by Q.')
                    break

            if received % max(args.every, 1) == 0:
                idx = saved
                rgb_name = f'{idx:06d}.png'
                depth_name = f'{idx:06d}.png'
                cv2.imwrite(str(rgb_dir / rgb_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(depth_dir / depth_name), depth)
                writer.writerow({
                    'frame': idx,
                    'unix_s': f'{time.time():.9f}',
                    'relative_s': f'{rel:.9f}',
                    'rgb_file': f'rgb/{rgb_name}',
                    'depth_file': f'depth/{depth_name}',
                })
                saved += 1

                if saved % 30 == 0:
                    print(f'saved={saved} | elapsed={rel:.1f}s | approx_save_fps={saved/max(rel,1e-6):.1f}')

            if args.seconds > 0 and rel >= args.seconds:
                break
    except KeyboardInterrupt:
        print('\nStopped by Ctrl+C.')
    finally:
        f.flush()
        f.close()
        cv2.destroyAllWindows()
        sock.close(0)
        ctx.term()

    metadata['frames_saved'] = saved
    metadata['recording_seconds'] = time.perf_counter() - t0
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    print('DONE')
    print('frames:', saved)
    print('bundle:', out)
    print('next: python g1/scripts/verify_offline_bundle.py --bundle', out)


if __name__ == '__main__':
    main()
