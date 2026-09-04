import csv, json, argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Replay a recorded bundle and visually verify AprilTag -> box geometry without FoundationPose.')
    p.add_argument('--bundle', required=True)
    p.add_argument('--every', type=int, default=1)
    p.add_argument('--save-video', action='store_true')
    return p.parse_args()


def make_detector(name):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, name))
    if hasattr(aruco, 'ArucoDetector'):
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        return lambda gray: detector.detectMarkers(gray)
    params = aruco.DetectorParameters_create()
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def estimate_tag_pose(bgr, K, detect_fn, wanted_id, tag_size_m):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect_fn(gray)
    if ids is None:
        return None, None
    ids = ids.reshape(-1)
    idx = np.where(ids == wanted_id)[0]
    if len(idx) == 0:
        return None, None
    c = np.asarray(corners[int(idx[0])], dtype=np.float32).reshape(4, 2)
    h = tag_size_m * 0.5
    obj = np.asarray([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], np.float32)
    flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=flag)
    if not ok:
        return None, c
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T, c


def box_corners(dims):
    hx, hy, hz = np.asarray(dims, dtype=float) * 0.5
    return np.asarray([
        [-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
        [-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz],
    ], dtype=np.float64)


EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def project(points, T, K):
    pc = (T[:3,:3] @ points.T).T + T[:3,3]
    if np.any(pc[:,2] <= 1e-6):
        return None
    uv = np.empty((len(pc),2), dtype=float)
    uv[:,0] = K[0,0] * pc[:,0] / pc[:,2] + K[0,2]
    uv[:,1] = K[1,1] * pc[:,1] / pc[:,2] + K[1,2]
    return uv


def main():
    args = parse_args()
    bundle = Path(args.bundle)
    K = np.loadtxt(bundle / 'cam_K.txt').reshape(3,3)
    cfg = json.loads((bundle / 'apriltag_config.json').read_text())
    T_tag_box = np.asarray(cfg['T_tag_box'], dtype=float).reshape(4,4)
    dims = np.asarray(cfg['box_dimensions_m'], dtype=float)
    wanted_id = int(cfg['marker_id'])
    tag_size_m = float(cfg['marker_size_m'])
    detect_fn = make_detector(cfg['dictionary'])
    corners_box = box_corners(dims)
    rows = list(csv.DictReader((bundle / 'timestamps.csv').open()))

    out_dir = bundle / 'apriltag_box_sanity'
    out_dir.mkdir(exist_ok=True)
    video = None

    print('OFFLINE APRILTAG -> BOX SANITY CHECK')
    print('Yellow wireframe is inferred ONLY from AprilTag + T_tag_box.')
    print('It should coincide with the physical 40x30x30 cm box.')
    print('S saves current overlay | SPACE pauses | Q quits')

    paused = False
    saved = 0
    for i, row in enumerate(rows):
        if i % max(args.every,1) != 0:
            continue
        bgr = cv2.imread(str(bundle / row['rgb_file']), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        T_cam_tag, tag_corners = estimate_tag_pose(bgr, K, detect_fn, wanted_id, tag_size_m)
        vis = bgr.copy()
        T_cam_box = None
        if tag_corners is not None:
            poly = np.round(tag_corners).astype(np.int32).reshape(-1,1,2)
            cv2.polylines(vis, [poly], True, (255,255,0), 2)

        if T_cam_tag is not None:
            T_cam_box = T_cam_tag @ T_tag_box
            uv = project(corners_box, T_cam_box, K)
            if uv is not None:
                p = np.round(uv).astype(np.int32)
                for a,b in EDGES:
                    cv2.line(vis, tuple(p[a]), tuple(p[b]), (0,255,255), 2, cv2.LINE_AA)
            rvec, _ = cv2.Rodrigues(T_cam_tag[:3,:3])
            cv2.drawFrameAxes(vis, K, None, rvec, T_cam_tag[:3,3], tag_size_m*0.7, 2)
            msg = f'frame {row["frame"]} | tag={np.linalg.norm(T_cam_tag[:3,3]):.3f}m | box={np.linalg.norm(T_cam_box[:3,3]):.3f}m'
        else:
            msg = f'frame {row["frame"]} | tag {wanted_id} NOT DETECTED'

        cv2.putText(vis, msg, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,0), 2)
        cv2.putText(vis, 'Yellow box must match real box BEFORE FoundationPose comparison',
                    (10,54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)

        if args.save_video and video is None:
            h,w = vis.shape[:2]
            video = cv2.VideoWriter(str(out_dir / 'sanity_overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (w,h))
        if video is not None:
            video.write(vis)

        while True:
            cv2.imshow('offline AprilTag -> box sanity', vis)
            key = cv2.waitKey(0 if paused else 30) & 0xFF
            if key == ord('q'):
                if video is not None:
                    video.release()
                cv2.destroyAllWindows()
                return
            if key == ord(' '):
                paused = not paused
                if not paused:
                    break
                continue
            if key == ord('s'):
                saved += 1
                cv2.imwrite(str(out_dir / f'sample_{saved:03d}.png'), vis)
                if T_cam_tag is not None:
                    np.savetxt(out_dir / f'sample_{saved:03d}_T_cam_tag.txt', T_cam_tag)
                    np.savetxt(out_dir / f'sample_{saved:03d}_T_cam_box.txt', T_cam_box)
                print('saved sample', saved)
            if not paused:
                break

    if video is not None:
        video.release()
    cv2.destroyAllWindows()
    print('Replay finished. Saved overlays:', out_dir)


if __name__ == '__main__':
    main()
