import argparse, csv
from pathlib import Path

import cv2
import numpy as np


EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


def parse_args():
    p = argparse.ArgumentParser(
        description='Render a FoundationPose-only demo video from a recorded offline bundle.'
    )
    p.add_argument('--bundle', required=True)
    p.add_argument('--every', type=int, default=3, help='Write every Nth frame.')
    p.add_argument('--fps', type=float, default=15.0, help='Output video playback FPS.')
    p.add_argument('--box-dims', type=float, nargs=3, default=(0.4, 0.3, 0.3),
                   metavar=('X', 'Y', 'Z'), help='Box dimensions in metres.')
    p.add_argument('--axis-length', type=float, default=0.10, help='Pose-axis length in metres.')
    p.add_argument('--output', default='', help='Optional output path.')
    return p.parse_args()


def box_corners(dims):
    hx, hy, hz = np.asarray(dims, dtype=np.float64) * 0.5
    return np.asarray([
        [-hx,-hy,-hz], [ hx,-hy,-hz], [ hx, hy,-hz], [-hx, hy,-hz],
        [-hx,-hy, hz], [ hx,-hy, hz], [ hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float64)


def load_pose(row):
    T = np.eye(4, dtype=np.float64)
    for r in range(4):
        for c in range(4):
            T[r, c] = float(row[f't{r}{c}'])
    return T


def project(points_obj, T_cam_obj, K):
    pc = (T_cam_obj[:3, :3] @ points_obj.T).T + T_cam_obj[:3, 3]
    if np.any(pc[:, 2] <= 1e-6):
        return None
    uv = np.empty((len(pc), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    uv[:, 1] = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    return uv


def draw_wireframe(img, uv):
    if uv is None:
        return
    p = np.round(uv).astype(np.int32)
    for a, b in EDGES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), (0, 255, 0), 2, cv2.LINE_AA)


def draw_pose_axis(img, T_cam_obj, K, length):
    pts = np.asarray([
        [0.0, 0.0, 0.0],
        [length, 0.0, 0.0],
        [0.0, length, 0.0],
        [0.0, 0.0, length],
    ], dtype=np.float64)
    uv = project(pts, T_cam_obj, K)
    if uv is None:
        return
    p = np.round(uv).astype(np.int32)
    origin = tuple(p[0])
    cv2.line(img, origin, tuple(p[1]), (0, 0, 255), 3, cv2.LINE_AA)   # X
    cv2.line(img, origin, tuple(p[2]), (0, 255, 0), 3, cv2.LINE_AA)   # Y
    cv2.line(img, origin, tuple(p[3]), (255, 0, 0), 3, cv2.LINE_AA)   # Z


def source_fps_from_timestamps(timestamps):
    times = []
    for row in timestamps:
        try:
            times.append(float(row['relative_s']))
        except (KeyError, TypeError, ValueError):
            pass
    if len(times) < 2:
        return 0.0
    dts = np.diff(np.asarray(times, dtype=np.float64))
    dts = dts[dts > 1e-6]
    if len(dts) == 0:
        return 0.0
    return float(1.0 / np.median(dts))


def put_text(img, text, y, scale=0.52):
    cv2.putText(
        img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 255, 0), 2, cv2.LINE_AA,
    )


def main():
    args = parse_args()
    bundle = Path(args.bundle)

    K = np.loadtxt(bundle / 'cam_K.txt').reshape(3, 3)
    timestamps = list(csv.DictReader((bundle / 'timestamps.csv').open()))
    fp_csv = bundle / 'foundationpose_offline' / 'foundationpose_poses.csv'
    if not fp_csv.exists():
        raise RuntimeError(f'Missing {fp_csv}')

    fp_rows = {str(r['frame']): r for r in csv.DictReader(fp_csv.open())}
    box_pts = box_corners(args.box_dims)
    source_fps = source_fps_from_timestamps(timestamps)

    out_dir = bundle / 'offline_comparison'
    out_dir.mkdir(exist_ok=True)
    output = Path(args.output) if args.output else out_dir / 'foundationpose_clean.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)

    video = None
    written = 0
    last_index = max(len(timestamps) - 1, 0)

    for i, ts in enumerate(timestamps):
        if i % max(args.every, 1) != 0:
            continue

        frame_id = str(ts['frame'])
        row = fp_rows.get(frame_id)
        if row is None:
            continue

        bgr = cv2.imread(str(bundle / ts['rgb_file']), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        T_fp = load_pose(row)
        vis = bgr.copy()
        draw_wireframe(vis, project(box_pts, T_fp, K))
        draw_pose_axis(vis, T_fp, K, args.axis_length)

        try:
            track_ms = float(row.get('track_ms', 0.0))
        except (TypeError, ValueError):
            track_ms = 0.0
        try:
            distance_m = float(row.get('distance_m', np.linalg.norm(T_fp[:3, 3])))
        except (TypeError, ValueError):
            distance_m = float(np.linalg.norm(T_fp[:3, 3]))

        pose_fps = (1000.0 / track_ms) if track_ms > 1e-6 else 0.0
        put_text(vis, f'FoundationPose | {pose_fps:.1f} FPS | track {track_ms:.1f} ms', 28)
        put_text(vis, f'frame {i}/{last_index} | distance {distance_m:.3f} m | source {source_fps:.1f} FPS', 54, 0.48)

        if video is None:
            h, w = vis.shape[:2]
            video = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h)
            )
            if not video.isOpened():
                raise RuntimeError(f'Could not open video writer for {output}')

        video.write(vis)
        written += 1

    if video is not None:
        video.release()

    if written == 0:
        raise RuntimeError('No frames were written.')

    print(f'Wrote {written} frames')
    print(f'Source FPS: {source_fps:.2f}')
    print(f'Output playback FPS: {args.fps:.2f}')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
