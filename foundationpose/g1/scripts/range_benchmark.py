import os, sys, base64, time, csv, json, math
from collections import deque
from pathlib import Path

import cv2, zmq, msgpack, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from estimater import *

ENDPOINT = 'tcp://192.168.123.164:5555'
INIT = ROOT / 'g1/data/live_init'
OUT = ROOT / 'g1/results/range_benchmark'
K = np.loadtxt(INIT / 'cam_K.txt').reshape(3, 3)

TRACK_ITERS = 1
REGISTER_ITERS = 5
SAMPLE_FRAMES = 60
VIS_EVERY = 3

# Stationary-object repeatability heuristic only; not absolute ground-truth accuracy.
STABLE_STEP_TRANSLATION_MM = 20.0
STABLE_STEP_ROTATION_DEG = 10.0


def decode(payload):
    data = msgpack.unpackb(payload, raw=False)
    out = {}
    for k, v in data['images'].items():
        b = base64.b64decode(v) if isinstance(v, str) else v
        out[k] = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    return out


def depth_m(depth_u16):
    d = depth_u16.astype(np.float32)
    d *= 0.001
    d[(d < 0.001) | (d > 10.0)] = 0
    return d


def recv_rgbd(sock):
    while True:
        im = decode(sock.recv())
        if 'ego_view' in im and 'ego_view_depth' in im:
            return im['ego_view'], im['ego_view_depth']


def mask_ui(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    shown = bgr.copy()
    pts = []
    win = 'Initial mask: outline box ONCE at a good distance'

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
    print('Initial mask: left-click box; ENTER accept; R reset; ESC cancel')
    while True:
        frame = shown.copy()
        cv2.putText(frame, 'Draw ONCE | ENTER accept | R reset | ESC cancel',
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


def pose_metrics(poses):
    if len(poses) < 2:
        return dict(translation_spread_p95_mm=float('nan'),
                    rotation_spread_p95_deg=float('nan'),
                    step_translation_p95_mm=float('nan'),
                    step_rotation_p95_deg=float('nan'),
                    stable_heuristic=False)

    ts = np.asarray([p[:3, 3] for p in poses])
    t_ref = np.median(ts, axis=0)
    spread_t = np.linalg.norm(ts - t_ref[None, :], axis=1) * 1000.0
    R_ref = poses[0][:3, :3]
    spread_r = np.asarray([rotation_angle_deg(R_ref, p[:3, :3]) for p in poses])
    step_t = np.linalg.norm(np.diff(ts, axis=0), axis=1) * 1000.0
    step_r = np.asarray([
        rotation_angle_deg(poses[i - 1][:3, :3], poses[i][:3, :3])
        for i in range(1, len(poses))
    ])
    step_t_p95 = float(np.percentile(step_t, 95))
    step_r_p95 = float(np.percentile(step_r, 95))
    stable = (step_t_p95 <= STABLE_STEP_TRANSLATION_MM and
              step_r_p95 <= STABLE_STEP_ROTATION_DEG)
    return dict(
        translation_spread_p95_mm=float(np.percentile(spread_t, 95)),
        rotation_spread_p95_deg=float(np.percentile(spread_r, 95)),
        step_translation_p95_mm=step_t_p95,
        step_rotation_p95_deg=step_r_p95,
        stable_heuristic=bool(stable),
    )


def append_csv(path, row):
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def bbox_corners(bbox):
    lo, hi = bbox[0], bbox[1]
    return np.asarray([[x, y, z]
                       for x in (lo[0], hi[0])
                       for y in (lo[1], hi[1])
                       for z in (lo[2], hi[2])], dtype=np.float32)


def visible_corner_pct(center_pose, bbox, h, w):
    pts = bbox_corners(bbox)
    cam = (center_pose[:3, :3] @ pts.T).T + center_pose[:3, 3]
    good_z = cam[:, 2] > 1e-6
    uv = np.full((len(cam), 2), np.nan, dtype=np.float32)
    uv[good_z, 0] = K[0, 0] * cam[good_z, 0] / cam[good_z, 2] + K[0, 2]
    uv[good_z, 1] = K[1, 1] * cam[good_z, 1] / cam[good_z, 2] + K[1, 2]
    inside = good_z & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return 100.0 * float(np.count_nonzero(inside)) / len(pts)


def center_depth_support(depth, center_pose, radius=6):
    x, y, z = center_pose[:3, 3]
    if z <= 1e-6:
        return float('nan'), 0.0
    u = int(round(K[0, 0] * x / z + K[0, 2]))
    v = int(round(K[1, 1] * y / z + K[1, 2]))
    h, w = depth.shape
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return float('nan'), 0.0
    patch = depth[y0:y1, x0:x1]
    valid = patch > 0
    if not np.any(valid):
        return float('nan'), 0.0
    return float(np.median(patch[valid])), 100.0 * float(valid.mean())


def main():
    set_logging_format()
    set_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)

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

    live_csv = OUT / 'live_tracking.csv'
    summary_csv = OUT / 'range_results.csv'

    print('RANGE TRACKING BENCHMARK — single initial mask')
    print('1) Put box at a good initial distance and draw mask ONCE.')
    print('2) After registration, move box closer/farther; tracking continues live.')
    print(f'3) Stop box at a distance and press S: next {SAMPLE_FRAMES} frames become one stability sample.')
    print('Q quits. live_tracking.csv logs EVERY tracked frame after registration.')

    try:
        rgb0, depth_raw0 = recv_rgbd(sock)
        depth0 = depth_m(depth_raw0)
        mask = mask_ui(rgb0)
        if mask is None:
            print('Cancelled before registration.')
            return

        cv2.imwrite(str(OUT / 'initial_rgb.png'), cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT / 'initial_depth.png'), depth_raw0)
        cv2.imwrite(str(OUT / 'initial_mask.png'), mask.astype(np.uint8) * 255)

        t0 = time.perf_counter()
        pose = est.register(K=K, rgb=rgb0, depth=depth0, ob_mask=mask,
                            iteration=REGISTER_ITERS)
        register_ms = (time.perf_counter() - t0) * 1000.0
        if not np.isfinite(pose).all():
            raise RuntimeError('Initial registration returned non-finite pose')
        np.savetxt(OUT / 'initial_pose.txt', pose)
        print(f'Initial registration OK: {register_ms:.1f} ms')
        print('Now DO NOT redraw the mask. Move box through the ranges you want to test.')

        frame_id = 0
        sample_id = 0
        prev_pose = pose.copy()
        last_frame_time = time.perf_counter()
        sample_buffer = None

        while True:
            rgb, depth_raw = recv_rgbd(sock)
            depth = depth_m(depth_raw)
            t_track = time.perf_counter()
            pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=TRACK_ITERS)
            track_ms = (time.perf_counter() - t_track) * 1000.0
            now = time.perf_counter()
            loop_fps = 1.0 / max(now - last_frame_time, 1e-9)
            last_frame_time = now
            frame_id += 1

            center_pose = pose @ center_tf
            distance_m = float(np.linalg.norm(center_pose[:3, 3]))
            z_m = float(center_pose[2, 3])
            step_t_mm = float(np.linalg.norm(pose[:3, 3] - prev_pose[:3, 3]) * 1000.0)
            step_r_deg = float(rotation_angle_deg(prev_pose[:3, :3], pose[:3, :3]))
            prev_pose = pose.copy()
            visible_pct = visible_corner_pct(center_pose, bbox, rgb.shape[0], rgb.shape[1])
            center_depth_m, center_depth_valid_pct = center_depth_support(depth, center_pose)

            live_row = {
                'frame': frame_id,
                'timestamp': time.time(),
                'distance_m_from_pose': distance_m,
                'pose_center_z_m': z_m,
                'pose_x_m': float(center_pose[0, 3]),
                'pose_y_m': float(center_pose[1, 3]),
                'track_ms': track_ms,
                'loop_fps': loop_fps,
                'step_translation_mm': step_t_mm,
                'step_rotation_deg': step_r_deg,
                'bbox_corner_visible_pct': visible_pct,
                'center_depth_m': center_depth_m,
                'center_depth_valid_pct': center_depth_valid_pct,
            }
            append_csv(live_csv, live_row)

            if sample_buffer is not None:
                sample_buffer.append((pose.copy(), live_row.copy()))
                if len(sample_buffer) >= SAMPLE_FRAMES:
                    sample_id += 1
                    poses = [x[0] for x in sample_buffer]
                    rows = [x[1] for x in sample_buffer]
                    pm = pose_metrics(poses)
                    summary = {
                        'sample': sample_id,
                        'timestamp': time.time(),
                        'frames': len(rows),
                        'distance_m_mean': float(np.mean([r['distance_m_from_pose'] for r in rows])),
                        'distance_m_std': float(np.std([r['distance_m_from_pose'] for r in rows])),
                        'mean_track_ms': float(np.mean([r['track_ms'] for r in rows])),
                        'p95_track_ms': float(np.percentile([r['track_ms'] for r in rows], 95)),
                        'mean_loop_fps': float(np.mean([r['loop_fps'] for r in rows])),
                        'bbox_corner_visible_pct_mean': float(np.mean([r['bbox_corner_visible_pct'] for r in rows])),
                        'center_depth_valid_pct_mean': float(np.mean([r['center_depth_valid_pct'] for r in rows])),
                        **pm,
                    }
                    append_csv(summary_csv, summary)
                    sample_dir = OUT / f'sample_{sample_id:03d}'
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    (sample_dir / 'metrics.json').write_text(json.dumps(summary, indent=2))
                    np.savetxt(sample_dir / 'latest_pose.txt', poses[-1])
                    print(f'SAMPLE {sample_id}: distance={summary["distance_m_mean"]:.3f} m | '
                          f'spread={pm["translation_spread_p95_mm"]:.1f} mm / '
                          f'{pm["rotation_spread_p95_deg"]:.2f} deg | '
                          f'step p95={pm["step_translation_p95_mm"]:.1f} mm / '
                          f'{pm["step_rotation_p95_deg"]:.2f} deg | '
                          f'visible={summary["bbox_corner_visible_pct_mean"]:.0f}% | '
                          f'stable={pm["stable_heuristic"]}')
                    sample_buffer = None

            if frame_id % VIS_EVERY == 0:
                vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                show = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                status = 'SAMPLING' if sample_buffer is not None else 'S: sample'
                cv2.putText(show,
                            f'{distance_m:.2f} m | {track_ms:.1f} ms | step {step_t_mm:.1f} mm | visible {visible_pct:.0f}% | {status}',
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
                cv2.imshow('FoundationPose range benchmark', show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and sample_buffer is None:
                sample_buffer = []
                print(f'Collecting next {SAMPLE_FRAMES} frames at current distance. Keep box stationary...')

    finally:
        cv2.destroyAllWindows()
        sock.close(0)
        ctx.term()
        print('Saved live log:', live_csv)
        print('Saved sample summaries:', summary_csv)


if __name__ == '__main__':
    main()
