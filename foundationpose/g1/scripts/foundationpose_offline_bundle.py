import os, sys, time, csv, argparse
from pathlib import Path
from collections import deque

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from estimater import *

TRACK_ITERS = 1
REGISTER_ITERS = 5
VIS_EVERY = 3


def parse_args():
    p = argparse.ArgumentParser(description='Run FoundationPose offline on a recorded G1 RGB-D bundle.')
    p.add_argument('--bundle', required=True)
    p.add_argument('--max-frames', type=int, default=0, help='0 = all frames')
    p.add_argument('--vis-every', type=int, default=VIS_EVERY)
    p.add_argument(
        '--mask-file', default='',
        help=(
            'Optional binary mask for first frame. If omitted, uses/saves '
            '<output-dir>/initial_mask.png and opens the drawing UI when missing.'
        ),
    )
    p.add_argument(
        '--output-dir', default='',
        help=(
            'Optional output directory. Default: <bundle>/foundationpose_offline. '
            'For multi-instance replay use separate directories, e.g. '
            '<bundle>/foundationpose_instances/carried and .../support.'
        ),
    )
    p.add_argument(
        '--window-name', default='FoundationPose OFFLINE bundle replay',
        help='Visualization window title.',
    )
    p.add_argument(
        '--no-display', action='store_true',
        help='Disable cv2.imshow while still writing pose CSV/latest pose.',
    )
    return p.parse_args()


def depth_m(d):
    d = d.astype(np.float32) * 0.001
    d[(d < 0.001) | (d > 10.0)] = 0
    return d


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f'Failed to read RGB: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_ui(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    shown = bgr.copy()
    pts = []
    win = 'Offline FoundationPose: draw FIRST-FRAME box mask'

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
        cv2.putText(frame, 'ENTER accept | R reset | ESC cancel', (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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


def load_binary_mask(path, expected_shape):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f'Failed to read mask: {path}')
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != tuple(expected_shape):
        raise RuntimeError(
            f'Mask shape {mask.shape[:2]} does not match RGB shape {tuple(expected_shape)}: {path}'
        )
    mask = mask > 0
    if int(mask.sum()) < 20:
        raise RuntimeError(f'Mask is empty or too small: {path}')
    return mask


def fps_from_times(ts):
    if len(ts) < 2:
        return 0.0
    dt = ts[-1] - ts[0]
    return (len(ts) - 1) / dt if dt > 0 else 0.0


def main():
    args = parse_args()
    set_logging_format()
    set_seed(0)

    bundle = Path(args.bundle)
    K = np.loadtxt(bundle / 'cam_K.txt').reshape(3, 3)
    rows = list(csv.DictReader((bundle / 'timestamps.csv').open()))
    if args.max_frames > 0:
        rows = rows[:args.max_frames]
    if len(rows) < 2:
        raise RuntimeError(f'Need at least 2 RGB-D frames; found {len(rows)}')

    mesh_path = bundle / 'box.obj'
    if not mesh_path.exists():
        mesh_path = ROOT / 'box.obj'
    mesh = trimesh.load(mesh_path)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    center_tf = np.linalg.inv(to_origin)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    out = Path(args.output_dir) if args.output_dir else bundle / 'foundationpose_offline'
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    default_mask_file = out / 'initial_mask.png'
    pose_csv = out / 'foundationpose_poses.csv'

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=str(out),
        debug=0,
        glctx=dr.RasterizeCudaContext(),
    )

    rgb0 = load_rgb(bundle / rows[0]['rgb_file'])
    dep0_raw = cv2.imread(str(bundle / rows[0]['depth_file']), cv2.IMREAD_UNCHANGED)
    dep0 = depth_m(dep0_raw)

    if args.mask_file:
        mask_path = Path(args.mask_file)
        if not mask_path.is_absolute():
            mask_path = ROOT / mask_path
        mask = load_binary_mask(mask_path, rgb0.shape[:2])
        cv2.imwrite(str(default_mask_file), mask.astype(np.uint8) * 255)
        print('Using supplied mask:', mask_path)
        print('Saved instance mask copy:', default_mask_file)
    elif default_mask_file.exists():
        mask = load_binary_mask(default_mask_file, rgb0.shape[:2])
        print('Using saved mask:', default_mask_file)
    else:
        if args.no_display:
            raise RuntimeError(
                'No mask available. With --no-display, provide --mask-file or pre-create '
                f'{default_mask_file}'
            )
        mask = mask_ui(rgb0)
        if mask is None:
            print('Cancelled.')
            return
        cv2.imwrite(str(default_mask_file), mask.astype(np.uint8) * 255)
        print('Saved mask:', default_mask_file)

    print(f'OFFLINE FOUNDATIONPOSE | frames={len(rows)} | register={REGISTER_ITERS} | track={TRACK_ITERS}')
    print('output:', out)
    t0 = time.perf_counter()
    pose = est.register(K=K, rgb=rgb0, depth=dep0, ob_mask=mask, iteration=REGISTER_ITERS)
    print(f'Registration OK: {(time.perf_counter()-t0)*1000.0:.1f} ms')

    fieldnames = ['frame', 'relative_s', 'track_ms', 'distance_m', 'center_z_m'] + [f't{r}{c}' for r in range(4) for c in range(4)]
    f = pose_csv.open('w', newline='')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    pose_times = deque(maxlen=120)
    track_times = deque(maxlen=120)
    processed = 0
    start = time.perf_counter()

    try:
        for i, row in enumerate(rows):
            if i == 0:
                track_ms = 0.0
            else:
                rgb = load_rgb(bundle / row['rgb_file'])
                dep_raw = cv2.imread(str(bundle / row['depth_file']), cv2.IMREAD_UNCHANGED)
                dep = depth_m(dep_raw)
                q0 = time.perf_counter()
                pose = est.track_one(rgb=rgb, depth=dep, K=K, iteration=TRACK_ITERS)
                track_ms = (time.perf_counter() - q0) * 1000.0
                pose_times.append(time.perf_counter())
                track_times.append(track_ms)

            if i == 0:
                rgb = rgb0

            cp = pose @ center_tf
            distance_m = float(np.linalg.norm(cp[:3, 3]))
            center_z_m = float(cp[2, 3])
            outrow = {
                'frame': row['frame'],
                'relative_s': row['relative_s'],
                'track_ms': track_ms,
                'distance_m': distance_m,
                'center_z_m': center_z_m,
            }
            for r in range(4):
                for c in range(4):
                    outrow[f't{r}{c}'] = pose[r, c]
            writer.writerow(outrow)
            processed += 1

            if not args.no_display and i % max(args.vis_every, 1) == 0:
                vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=cp, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=cp, scale=0.1, K=K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                show = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                txt = f'FP frame {i}/{len(rows)-1} | d={distance_m:.3f}m | track={track_ms:.1f}ms'
                cv2.putText(show, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (0, 255, 0), 2)
                cv2.imshow(args.window_name, show)

            if not args.no_display and cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stopped by Q.')
                break
            if i > 0 and i % 100 == 0:
                print(f'frame={i} | recent pose FPS={fps_from_times(pose_times):.2f} | track={track_ms:.1f} ms')
    finally:
        f.flush()
        f.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    mean_track = float(np.mean(track_times)) if track_times else 0.0
    np.savetxt(out / 'latest_pose.txt', pose)
    print(f'DONE | processed={processed} | wall FPS={processed/max(elapsed,1e-6):.2f} | mean recent track={mean_track:.1f} ms')
    print('saved:', pose_csv)
    print('latest:', out / 'latest_pose.txt')


if __name__ == '__main__':
    main()
