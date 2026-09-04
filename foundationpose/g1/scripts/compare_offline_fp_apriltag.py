import argparse, csv, json, math
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
        description='Compare FoundationPose against AprilTag-derived box pose on one recorded offline bundle.'
    )
    p.add_argument('--bundle', required=True)
    p.add_argument('--every', type=int, default=1, help='Visualize every Nth frame.')
    p.add_argument('--max-frames', type=int, default=0, help='0 = all frames.')
    p.add_argument('--save-video', action='store_true')
    p.add_argument(
        '--clean-video',
        action='store_true',
        help=(
            'Render only the FoundationPose box overlay. AprilTag/reference overlays, '
            'axes, and comparison metrics are hidden, while evaluation is still computed.'
        ),
    )
    return p.parse_args()


def box_corners(dims):
    hx, hy, hz = np.asarray(dims, dtype=float) * 0.5
    return np.asarray([
        [-hx,-hy,-hz], [ hx,-hy,-hz], [ hx, hy,-hz], [-hx, hy,-hz],
        [-hx,-hy, hz], [ hx,-hy, hz], [ hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float64)


def project(points_obj, T_cam_obj, K):
    pc = (T_cam_obj[:3, :3] @ points_obj.T).T + T_cam_obj[:3, 3]
    if np.any(pc[:, 2] <= 1e-6):
        return None
    uv = np.empty((len(pc), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    uv[:, 1] = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    return uv


def draw_wireframe(img, uv, color, thickness=2):
    if uv is None:
        return
    p = np.round(uv).astype(np.int32)
    for a, b in EDGES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), color, thickness, cv2.LINE_AA)


def rot_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c],
    ], dtype=np.float64)


def rot180_about_yz_axis(phi_deg):
    # 180-degree rotation around unit axis u=(0, cos(phi), sin(phi)).
    a = math.radians(phi_deg)
    u = np.asarray([0.0, math.cos(a), math.sin(a)], dtype=np.float64)
    return 2.0 * np.outer(u, u) - np.eye(3)


# Proper rotational symmetry group of a 40x30x30 cuboid (D4, 8 elements):
# four rotations around the long X axis plus four 180-degree flips around axes
# lying in the square Y-Z cross-section. These all map the untextured cuboid
# geometry onto itself.
BOX_SYMMETRIES = [rx(a) for a in (0, 90, 180, 270)] + [
    rot180_about_yz_axis(a) for a in (0, 45, 90, 135)
]


def symmetry_rotation_error_deg(R_fp, R_tag_box):
    return min(rot_angle_deg(R_fp, R_tag_box @ S) for S in BOX_SYMMETRIES)


def load_fp_pose(row):
    T = np.eye(4, dtype=np.float64)
    for r in range(4):
        for c in range(4):
            T[r, c] = float(row[f't{r}{c}'])
    return T


def load_tag_pose(row):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [float(row['tx_m']), float(row['ty_m']), float(row['tz_m'])]
    for r in range(3):
        for c in range(3):
            T[r, c] = float(row[f'r{r}{c}'])
    return T


def percentile_or_none(values, q):
    return float(np.percentile(values, q)) if values else None


def main():
    args = parse_args()
    bundle = Path(args.bundle)

    K = np.loadtxt(bundle / 'cam_K.txt').reshape(3, 3)
    cfg = json.loads((bundle / 'apriltag_config.json').read_text())
    T_tag_box = np.asarray(cfg['T_tag_box'], dtype=np.float64).reshape(4, 4)
    dims = np.asarray(cfg['box_dimensions_m'], dtype=np.float64)
    box_pts = box_corners(dims)
    tag_half = float(cfg['marker_size_m']) * 0.5
    tag_pts = np.asarray([
        [-tag_half,  tag_half, 0.0],
        [ tag_half,  tag_half, 0.0],
        [ tag_half, -tag_half, 0.0],
        [-tag_half, -tag_half, 0.0],
    ], dtype=np.float64)

    timestamps = list(csv.DictReader((bundle / 'timestamps.csv').open()))
    if args.max_frames > 0:
        timestamps = timestamps[:args.max_frames]

    fp_csv = bundle / 'foundationpose_offline' / 'foundationpose_poses.csv'
    tag_csv = bundle / 'apriltag_poses.csv'
    if not fp_csv.exists():
        raise RuntimeError(
            f'Missing {fp_csv}. Run foundationpose_offline_bundle.py on this bundle first.'
        )
    if not tag_csv.exists():
        raise RuntimeError(
            f'Missing {tag_csv}. Run verify_offline_bundle.py on this bundle first.'
        )

    fp_rows = {str(r['frame']): r for r in csv.DictReader(fp_csv.open())}
    tag_rows = {str(r['frame']): r for r in csv.DictReader(tag_csv.open())}

    out_dir = bundle / 'offline_comparison'
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / 'foundationpose_vs_apriltag.csv'
    out_json = out_dir / 'summary.json'

    fields = [
        'frame', 'relative_s', 'tag_detected',
        'translation_error_mm', 'distance_error_mm',
        'rotation_error_raw_deg', 'rotation_error_symmetry_deg',
        'fp_distance_m', 'apriltag_box_distance_m',
    ]
    fout = out_csv.open('w', newline='')
    writer = csv.DictWriter(fout, fieldnames=fields)
    writer.writeheader()

    trans_errs = []
    dist_errs = []
    rot_raw = []
    rot_sym = []
    compared = 0
    saved = 0
    video = None
    video_path = None
    paused = False

    print('OFFLINE FOUNDATIONPOSE vs APRILTAG COMPARISON')
    print('Recorded files only: NO robot, NO camera stream, NO ZMQ.')
    if args.clean_video:
        print('Clean visualization: FoundationPose overlay only; AprilTag is used only for evaluation.')
    else:
        print('Green = FoundationPose | Yellow = AprilTag-derived box | Cyan = AprilTag')
    print('Rotation reports raw SO(3) error and full geometry-symmetry-aware error for the 40x30x30 box.')
    print('SPACE pause | S save overlay | Q quit')

    try:
        for i, ts in enumerate(timestamps):
            frame_id = str(ts['frame'])
            fp_row = fp_rows.get(frame_id)
            if fp_row is None:
                continue

            bgr = cv2.imread(str(bundle / ts['rgb_file']), cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            T_fp = load_fp_pose(fp_row)
            fp_d = float(np.linalg.norm(T_fp[:3, 3]))

            tag_row = tag_rows.get(frame_id)
            tag_detected = (
                tag_row is not None and
                str(tag_row.get('detected', '')).strip().lower() in ('true', '1', 'yes')
            )

            row_out = {
                'frame': frame_id,
                'relative_s': ts['relative_s'],
                'tag_detected': tag_detected,
                'fp_distance_m': fp_d,
            }

            T_tag = None
            T_tag_box_cam = None
            trans_mm = dist_mm = raw_deg = sym_deg = None

            if tag_detected:
                T_tag = load_tag_pose(tag_row)
                T_tag_box_cam = T_tag @ T_tag_box
                tag_box_d = float(np.linalg.norm(T_tag_box_cam[:3, 3]))
                trans_mm = float(np.linalg.norm(T_fp[:3, 3] - T_tag_box_cam[:3, 3]) * 1000.0)
                dist_mm = float(abs(fp_d - tag_box_d) * 1000.0)
                raw_deg = rot_angle_deg(T_fp[:3, :3], T_tag_box_cam[:3, :3])
                sym_deg = symmetry_rotation_error_deg(T_fp[:3, :3], T_tag_box_cam[:3, :3])

                trans_errs.append(trans_mm)
                dist_errs.append(dist_mm)
                rot_raw.append(raw_deg)
                rot_sym.append(sym_deg)
                compared += 1

                row_out.update({
                    'translation_error_mm': trans_mm,
                    'distance_error_mm': dist_mm,
                    'rotation_error_raw_deg': raw_deg,
                    'rotation_error_symmetry_deg': sym_deg,
                    'apriltag_box_distance_m': tag_box_d,
                })
            writer.writerow(row_out)

            if i % max(args.every, 1) != 0:
                continue

            vis = bgr.copy()
            draw_wireframe(vis, project(box_pts, T_fp, K), (0, 255, 0), 2)

            if args.clean_video:
                cv2.putText(
                    vis,
                    'FoundationPose tracking',
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                if T_tag_box_cam is not None:
                    draw_wireframe(vis, project(box_pts, T_tag_box_cam, K), (0, 255, 255), 2)
                    tag_uv = project(tag_pts, T_tag, K)
                    if tag_uv is not None:
                        p = np.round(tag_uv).astype(np.int32)
                        cv2.polylines(vis, [p.reshape(-1, 1, 2)], True, (255, 255, 0), 2, cv2.LINE_AA)
                    rvec, _ = cv2.Rodrigues(T_tag[:3, :3])
                    cv2.drawFrameAxes(vis, K, None, rvec, T_tag[:3, 3], float(cfg['marker_size_m']) * 0.7, 2)

                    line1 = f'frame {frame_id} | trans={trans_mm:.1f} mm | distance={dist_mm:.1f} mm'
                    line2 = f'rotation raw={raw_deg:.1f} deg | symmetry-aware={sym_deg:.1f} deg'
                else:
                    line1 = f'frame {frame_id} | AprilTag not detected'
                    line2 = 'FoundationPose still shown in green'

                cv2.putText(vis, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)
                cv2.putText(vis, line2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 2)
                cv2.putText(vis, 'green=FP  yellow=tag-box  cyan=tag | SPACE pause S save Q quit',
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255,255,255), 1)

            if args.save_video and video is None:
                h, w = vis.shape[:2]
                video_name = 'foundationpose_clean.mp4' if args.clean_video else 'comparison_overlay.mp4'
                video_path = out_dir / video_name
                video = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (w, h)
                )
            if video is not None:
                video.write(vis)

            window_name = 'FoundationPose tracking' if args.clean_video else 'Offline FP vs AprilTag'
            while True:
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(0 if paused else 25) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
                if key == ord(' '):
                    paused = not paused
                    if not paused:
                        break
                    continue
                if key == ord('s'):
                    saved += 1
                    prefix = 'clean_overlay' if args.clean_video else 'overlay'
                    cv2.imwrite(str(out_dir / f'{prefix}_{saved:03d}.png'), vis)
                    print('saved overlay', saved)
                if not paused:
                    break

    except KeyboardInterrupt:
        print('Stopped by user.')
    finally:
        fout.flush()
        fout.close()
        if video is not None:
            video.release()
        cv2.destroyAllWindows()

    summary = {
        'frames_in_bundle_considered': len(timestamps),
        'frames_compared_with_both_poses': compared,
        'translation_error_mm': {
            'mean': float(np.mean(trans_errs)) if trans_errs else None,
            'median': float(np.median(trans_errs)) if trans_errs else None,
            'p95': percentile_or_none(trans_errs, 95),
        },
        'distance_error_mm': {
            'mean': float(np.mean(dist_errs)) if dist_errs else None,
            'median': float(np.median(dist_errs)) if dist_errs else None,
            'p95': percentile_or_none(dist_errs, 95),
        },
        'rotation_error_raw_deg': {
            'mean': float(np.mean(rot_raw)) if rot_raw else None,
            'median': float(np.median(rot_raw)) if rot_raw else None,
            'p95': percentile_or_none(rot_raw, 95),
        },
        'rotation_error_symmetry_deg': {
            'mean': float(np.mean(rot_sym)) if rot_sym else None,
            'median': float(np.median(rot_sym)) if rot_sym else None,
            'p95': percentile_or_none(rot_sym, 95),
        },
        'symmetry_note': 'The untextured 40x30x30 cuboid has 8 proper rotational symmetries (D4), including 180-degree flips as well as 90-degree rotations around the long axis.',
    }
    out_json.write_text(json.dumps(summary, indent=2))

    print('COMPARISON SUMMARY')
    print('frames compared:', compared)
    if compared:
        print(f'translation error mm: mean={np.mean(trans_errs):.2f} median={np.median(trans_errs):.2f} p95={np.percentile(trans_errs,95):.2f}')
        print(f'distance error mm: mean={np.mean(dist_errs):.2f} median={np.median(dist_errs):.2f} p95={np.percentile(dist_errs,95):.2f}')
        print(f'rotation raw deg: mean={np.mean(rot_raw):.2f} median={np.median(rot_raw):.2f} p95={np.percentile(rot_raw,95):.2f}')
        print(f'rotation symmetry-aware deg: mean={np.mean(rot_sym):.2f} median={np.median(rot_sym):.2f} p95={np.percentile(rot_sym,95):.2f}')
    print('saved:', out_csv)
    print('saved:', out_json)
    if video_path is not None:
        print('saved:', video_path)


if __name__ == '__main__':
    main()
