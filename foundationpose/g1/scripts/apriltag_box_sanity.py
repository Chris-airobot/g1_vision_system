import base64, json, time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = 'tcp://192.168.123.164:5555'
INIT = ROOT / 'g1/data/live_init'
CFG_PATH = ROOT / 'g1/config/apriltag_gt_placeholder.json'
OUT = ROOT / 'g1/results/apriltag_box_sanity'
K = np.loadtxt(INIT / 'cam_K.txt').reshape(3, 3)


def decode(payload):
    data = msgpack.unpackb(payload, raw=False)
    out = {}
    for k, v in data['images'].items():
        b = base64.b64decode(v) if isinstance(v, str) else v
        out[k] = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    return out


def recv_rgb(sock):
    while True:
        im = decode(sock.recv())
        if 'ego_view' in im:
            return im['ego_view']


def make_detector(dictionary_name):
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('cv2.aruco unavailable; use an OpenCV build with aruco support')
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    if hasattr(aruco, 'ArucoDetector'):
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        return lambda gray: detector.detectMarkers(gray)
    params = aruco.DetectorParameters_create()
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def estimate_tag_pose(rgb, detect_fn, wanted_id, tag_size_m):
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


def project_points(points_box, T_cam_box):
    pts_cam = (T_cam_box[:3, :3] @ points_box.T).T + T_cam_box[:3, 3]
    if np.any(pts_cam[:, 2] <= 1e-6):
        return None
    uv = np.empty((len(pts_cam), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    uv[:, 1] = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    return uv


def draw_box_wireframe(bgr, uv):
    if uv is None:
        return bgr
    p = np.round(uv).astype(np.int32)
    for a, b in EDGES:
        cv2.line(bgr, tuple(p[a]), tuple(p[b]), (0, 255, 255), 2, cv2.LINE_AA)
    return bgr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CFG_PATH.read_text())
    wanted_id = int(cfg['marker_id'])
    tag_size_m = float(cfg['marker_size_m'])
    T_tag_box = np.asarray(cfg['T_tag_box'], dtype=np.float64).reshape(4, 4)
    dims = np.asarray(cfg['box_dimensions_m'], dtype=np.float64)
    corners_box = box_corners(dims)
    detect_fn = make_detector(cfg['dictionary'])

    print('APRILTAG -> BOX SANITY CHECK (NO FOUNDATIONPOSE)')
    print(f"family={cfg['dictionary']} id={wanted_id} size={tag_size_m:.3f}m")
    print('Yellow wireframe = box pose inferred only from AprilTag + T_tag_box.')
    print('Goal: wireframe must visually coincide with the real 40x30x30 cm box.')
    print('S saves current RGB + poses; Q quits.')

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, '')
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(ENDPOINT)

    save_id = 0
    try:
        while True:
            rgb = recv_rgb(sock)
            T_cam_tag, tag_corners = estimate_tag_pose(rgb, detect_fn, wanted_id, tag_size_m)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            T_cam_box = None

            if tag_corners is not None:
                poly = np.round(tag_corners).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(bgr, [poly], True, (255, 255, 0), 2)

            if T_cam_tag is not None:
                T_cam_box = T_cam_tag @ T_tag_box
                uv = project_points(corners_box, T_cam_box)
                draw_box_wireframe(bgr, uv)
                rvec, _ = cv2.Rodrigues(T_cam_tag[:3, :3])
                cv2.drawFrameAxes(bgr, K, None, rvec, T_cam_tag[:3, 3], tag_size_m * 0.7, 2)
                tag_d = float(np.linalg.norm(T_cam_tag[:3, 3]))
                box_d = float(np.linalg.norm(T_cam_box[:3, 3]))
                msg = f'TAG {tag_d:.3f}m | inferred BOX center {box_d:.3f}m'
            else:
                msg = f'AprilTag id {wanted_id} NOT DETECTED'

            cv2.putText(bgr, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)
            cv2.putText(bgr, 'Yellow box must match physical box before comparing FoundationPose',
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            cv2.imshow('AprilTag -> box sanity', bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and T_cam_tag is not None:
                save_id += 1
                stem = OUT / f'sample_{save_id:03d}'
                cv2.imwrite(str(stem.with_suffix('.png')), bgr)
                np.savetxt(str(stem) + '_T_cam_tag.txt', T_cam_tag)
                np.savetxt(str(stem) + '_T_cam_box.txt', T_cam_box)
                print(f'Saved sample {save_id}: box distance={np.linalg.norm(T_cam_box[:3, 3]):.3f} m')
    finally:
        cv2.destroyAllWindows()
        sock.close(0)
        ctx.term()


if __name__ == '__main__':
    main()
