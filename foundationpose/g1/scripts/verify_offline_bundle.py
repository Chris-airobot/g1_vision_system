import csv, json, math, argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Verify a recorded G1 offline RGB-D bundle and extract AprilTag poses.')
    p.add_argument('--bundle', required=True)
    p.add_argument('--max-frames', type=int, default=0,
                   help='0 = verify all frames; otherwise stop after N.')
    return p.parse_args()


def make_detector(dictionary_name):
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('cv2.aruco unavailable. Use an OpenCV build with aruco support.')
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    if hasattr(aruco, 'ArucoDetector'):
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        return lambda gray: detector.detectMarkers(gray)
    params = aruco.DetectorParameters_create()
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def estimate_tag_pose(bgr, K, detect_fn, wanted_id, tag_size_m):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect_fn(gray)
    if ids is None:
        return None
    ids_flat = ids.reshape(-1)
    matches = np.where(ids_flat == wanted_id)[0]
    if len(matches) == 0:
        return None

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
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def main():
    args = parse_args()
    bundle = Path(args.bundle)
    meta = json.loads((bundle / 'metadata.json').read_text())
    K = np.loadtxt(bundle / 'cam_K.txt').reshape(3, 3)
    tag_cfg = json.loads((bundle / 'apriltag_config.json').read_text())
    detect_fn = make_detector(tag_cfg['dictionary'])
    wanted_id = int(tag_cfg['marker_id'])
    tag_size_m = float(tag_cfg['marker_size_m'])

    rows = list(csv.DictReader((bundle / 'timestamps.csv').open()))
    if args.max_frames > 0:
        rows = rows[:args.max_frames]

    if not rows:
        raise RuntimeError('No frames listed in timestamps.csv')

    pose_csv = bundle / 'apriltag_poses.csv'
    pose_f = pose_csv.open('w', newline='')
    fieldnames = ['frame','detected','tx_m','ty_m','tz_m','distance_m'] + [f'r{i}{j}' for i in range(3) for j in range(3)]
    writer = csv.DictWriter(pose_f, fieldnames=fieldnames)
    writer.writeheader()

    rgb_ok = 0
    depth_ok = 0
    shape_match = 0
    tag_found = 0
    depth_valid_fracs = []
    rel_times = []
    tag_distances = []

    first_shape = None
    for i, row in enumerate(rows):
        rgb_path = bundle / row['rgb_file']
        depth_path = bundle / row['depth_file']
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        if bgr is not None:
            rgb_ok += 1
        if depth is not None:
            depth_ok += 1
        if bgr is None or depth is None:
            writer.writerow({'frame': row['frame'], 'detected': False})
            continue

        if bgr.shape[:2] == depth.shape[:2]:
            shape_match += 1
        if first_shape is None:
            first_shape = (bgr.shape, depth.shape, str(depth.dtype))

        valid = (depth > 0)
        depth_valid_fracs.append(float(valid.mean()))
        rel_times.append(float(row['relative_s']))

        T = estimate_tag_pose(bgr, K, detect_fn, wanted_id, tag_size_m)
        if T is None:
            writer.writerow({'frame': row['frame'], 'detected': False})
        else:
            tag_found += 1
            t = T[:3, 3]
            d = float(np.linalg.norm(t))
            tag_distances.append(d)
            out = {
                'frame': row['frame'], 'detected': True,
                'tx_m': t[0], 'ty_m': t[1], 'tz_m': t[2], 'distance_m': d,
            }
            for a in range(3):
                for b in range(3):
                    out[f'r{a}{b}'] = T[a, b]
            writer.writerow(out)

    pose_f.close()

    n = len(rows)
    dt = np.diff(rel_times) if len(rel_times) > 1 else np.asarray([])
    report = {
        'frames_checked': n,
        'rgb_readable_pct': 100.0 * rgb_ok / n,
        'depth_readable_pct': 100.0 * depth_ok / n,
        'rgb_depth_same_shape_pct': 100.0 * shape_match / n,
        'mean_depth_valid_pct': 100.0 * float(np.mean(depth_valid_fracs)) if depth_valid_fracs else 0.0,
        'apriltag_detection_pct': 100.0 * tag_found / n,
        'apriltag_frames': tag_found,
        'mean_frame_interval_ms': 1000.0 * float(np.mean(dt)) if len(dt) else None,
        'p95_frame_interval_ms': 1000.0 * float(np.percentile(dt, 95)) if len(dt) else None,
        'approx_fps': float(1.0 / np.mean(dt)) if len(dt) and np.mean(dt) > 0 else None,
        'apriltag_distance_mean_m': float(np.mean(tag_distances)) if tag_distances else None,
        'apriltag_distance_min_m': float(np.min(tag_distances)) if tag_distances else None,
        'apriltag_distance_max_m': float(np.max(tag_distances)) if tag_distances else None,
        'first_rgb_depth_shape_dtype': first_shape,
        'K': K.tolist(),
        'marker_id': wanted_id,
        'marker_size_m': tag_size_m,
    }
    (bundle / 'verification_report.json').write_text(json.dumps(report, indent=2))

    print('OFFLINE BUNDLE VERIFICATION')
    print(f'frames checked: {n}')
    print(f'RGB readable: {report["rgb_readable_pct"]:.1f}%')
    print(f'depth readable: {report["depth_readable_pct"]:.1f}%')
    print(f'RGB/depth shape match: {report["rgb_depth_same_shape_pct"]:.1f}%')
    print(f'mean valid depth: {report["mean_depth_valid_pct"]:.1f}%')
    print(f'AprilTag {wanted_id} detected: {report["apriltag_detection_pct"]:.1f}%')
    if report['approx_fps'] is not None:
        print(f'capture FPS: {report["approx_fps"]:.2f} | p95 frame interval: {report["p95_frame_interval_ms"]:.1f} ms')
    if tag_distances:
        print(f'AprilTag camera distance: mean={report["apriltag_distance_mean_m"]:.3f} m, '
              f'range=[{report["apriltag_distance_min_m"]:.3f}, {report["apriltag_distance_max_m"]:.3f}] m')
    print('saved:', bundle / 'verification_report.json')
    print('saved:', pose_csv)

    critical_ok = (
        report['rgb_readable_pct'] == 100.0 and
        report['depth_readable_pct'] == 100.0 and
        report['rgb_depth_same_shape_pct'] == 100.0
    )
    if critical_ok:
        print('CORE DATA INTEGRITY: PASS')
    else:
        print('CORE DATA INTEGRITY: FAIL')

    if report['apriltag_detection_pct'] >= 80.0:
        print('APRILTAG COVERAGE: GOOD')
    elif report['apriltag_detection_pct'] >= 50.0:
        print('APRILTAG COVERAGE: USABLE BUT IMPROVE VISIBILITY')
    else:
        print('APRILTAG COVERAGE: LOW')


if __name__ == '__main__':
    main()
