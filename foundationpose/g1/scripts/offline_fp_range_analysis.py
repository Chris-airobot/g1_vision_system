import argparse, csv, json, math
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description='Estimate FoundationPose usable range from an offline bundle, using AprilTag only as the distance reference.'
    )
    p.add_argument('--bundle', required=True)
    p.add_argument('--bin-m', type=float, default=0.10, help='Distance-bin width in meters (default 0.10).')
    p.add_argument('--stationary-trans-mm', type=float, default=12.0,
                   help='Tag inter-frame translation threshold for quasi-stationary pairs.')
    p.add_argument('--stationary-rot-deg', type=float, default=6.0,
                   help='Tag inter-frame rotation threshold for quasi-stationary pairs.')
    p.add_argument('--fp-jump-mm', type=float, default=50.0,
                   help='Large FoundationPose inter-frame translation jump threshold.')
    p.add_argument('--fp-jump-rot-deg', type=float, default=20.0,
                   help='Large symmetry-aware FoundationPose rotation jump threshold.')
    return p.parse_args()


def load_pose4(row, prefix='t'):
    T = np.eye(4, dtype=np.float64)
    for r in range(4):
        for c in range(4):
            T[r, c] = float(row[f'{prefix}{r}{c}'])
    return T


def load_tag_pose(row):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [float(row['tx_m']), float(row['ty_m']), float(row['tz_m'])]
    for r in range(3):
        for c in range(3):
            T[r, c] = float(row[f'r{r}{c}'])
    return T


def rot_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)


def rot180_about_yz_axis(phi_deg):
    a = math.radians(phi_deg)
    u = np.asarray([0.0, math.cos(a), math.sin(a)], dtype=np.float64)
    return 2.0 * np.outer(u, u) - np.eye(3)


BOX_SYMMETRIES = [rx(a) for a in (0, 90, 180, 270)] + [
    rot180_about_yz_axis(a) for a in (0, 45, 90, 135)
]


def symmetry_rot_step_deg(Ra, Rb):
    return min(rot_angle_deg(Ra, Rb @ S) for S in BOX_SYMMETRIES)


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def fmt(x, nd=2):
    return '-' if x is None else f'{x:.{nd}f}'


def main():
    args = parse_args()
    bundle = Path(args.bundle)
    fp_csv = bundle / 'foundationpose_offline' / 'foundationpose_poses.csv'
    tag_csv = bundle / 'apriltag_poses.csv'

    if not fp_csv.exists():
        raise RuntimeError(f'Missing {fp_csv}')
    if not tag_csv.exists():
        raise RuntimeError(f'Missing {tag_csv}')

    fp_rows = {str(r['frame']): r for r in csv.DictReader(fp_csv.open())}
    tag_rows = {str(r['frame']): r for r in csv.DictReader(tag_csv.open())}
    common = sorted(set(fp_rows) & set(tag_rows), key=lambda x: int(x))

    samples = []
    prev = None
    for frame in common:
        tr = tag_rows[frame]
        detected = str(tr.get('detected', '')).strip().lower() in ('true', '1', 'yes')
        if not detected:
            prev = None
            continue

        fr = fp_rows[frame]
        Tfp = load_pose4(fr)
        Ttag = load_tag_pose(tr)
        tag_d = float(tr['distance_m'])
        track_ms = float(fr.get('track_ms', 0.0) or 0.0)

        item = {
            'frame': int(frame),
            'tag_distance_m': tag_d,
            'track_ms': track_ms,
            'Tfp': Tfp,
            'Ttag': Ttag,
            'has_prev': False,
            'tag_step_mm': None,
            'tag_step_rot_deg': None,
            'fp_step_mm': None,
            'fp_step_rot_sym_deg': None,
            'quasi_stationary': False,
            'fp_large_jump': False,
        }

        if prev is not None and item['frame'] == prev['frame'] + 1:
            tag_step_mm = float(np.linalg.norm(Ttag[:3,3] - prev['Ttag'][:3,3]) * 1000.0)
            tag_step_rot = rot_angle_deg(prev['Ttag'][:3,:3], Ttag[:3,:3])
            fp_step_mm = float(np.linalg.norm(Tfp[:3,3] - prev['Tfp'][:3,3]) * 1000.0)
            fp_step_rot = symmetry_rot_step_deg(prev['Tfp'][:3,:3], Tfp[:3,:3])
            quasi = tag_step_mm <= args.stationary_trans_mm and tag_step_rot <= args.stationary_rot_deg
            jump = fp_step_mm > args.fp_jump_mm or fp_step_rot > args.fp_jump_rot_deg
            item.update({
                'has_prev': True,
                'tag_step_mm': tag_step_mm,
                'tag_step_rot_deg': tag_step_rot,
                'fp_step_mm': fp_step_mm,
                'fp_step_rot_sym_deg': fp_step_rot,
                'quasi_stationary': quasi,
                'fp_large_jump': jump,
            })

        samples.append(item)
        prev = item

    if not samples:
        raise RuntimeError('No frames with both FoundationPose output and detected AprilTag.')

    dmin = min(s['tag_distance_m'] for s in samples)
    dmax = max(s['tag_distance_m'] for s in samples)
    lo = math.floor(dmin / args.bin_m) * args.bin_m
    hi = math.ceil(dmax / args.bin_m) * args.bin_m

    out_dir = bundle / 'range_analysis'
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / 'fp_stability_by_tag_distance.csv'
    out_json = out_dir / 'summary.json'

    fields = [
        'distance_bin_m', 'distance_center_m', 'frames', 'quasi_stationary_pairs',
        'track_ms_mean', 'track_ms_p95',
        'fp_step_mm_median_all', 'fp_step_mm_p95_all',
        'fp_step_rot_deg_median_all', 'fp_step_rot_deg_p95_all',
        'fp_step_mm_median_stationary', 'fp_step_mm_p95_stationary',
        'fp_step_rot_deg_median_stationary', 'fp_step_rot_deg_p95_stationary',
        'large_jump_rate_pct_all',
    ]

    rows_out = []
    b = lo
    while b < hi + 1e-9:
        e = b + args.bin_m
        xs = [s for s in samples if (s['tag_distance_m'] >= b and s['tag_distance_m'] < e)]
        if xs:
            pair = [s for s in xs if s['has_prev']]
            stat = [s for s in pair if s['quasi_stationary']]
            fp_step_all = [s['fp_step_mm'] for s in pair]
            fp_rot_all = [s['fp_step_rot_sym_deg'] for s in pair]
            fp_step_stat = [s['fp_step_mm'] for s in stat]
            fp_rot_stat = [s['fp_step_rot_sym_deg'] for s in stat]
            jumps = [s['fp_large_jump'] for s in pair]
            track = [s['track_ms'] for s in xs]

            row = {
                'distance_bin_m': f'{b:.2f}-{e:.2f}',
                'distance_center_m': b + 0.5 * args.bin_m,
                'frames': len(xs),
                'quasi_stationary_pairs': len(stat),
                'track_ms_mean': float(np.mean(track)),
                'track_ms_p95': pct(track, 95),
                'fp_step_mm_median_all': float(np.median(fp_step_all)) if fp_step_all else None,
                'fp_step_mm_p95_all': pct(fp_step_all, 95),
                'fp_step_rot_deg_median_all': float(np.median(fp_rot_all)) if fp_rot_all else None,
                'fp_step_rot_deg_p95_all': pct(fp_rot_all, 95),
                'fp_step_mm_median_stationary': float(np.median(fp_step_stat)) if fp_step_stat else None,
                'fp_step_mm_p95_stationary': pct(fp_step_stat, 95),
                'fp_step_rot_deg_median_stationary': float(np.median(fp_rot_stat)) if fp_rot_stat else None,
                'fp_step_rot_deg_p95_stationary': pct(fp_rot_stat, 95),
                'large_jump_rate_pct_all': 100.0 * sum(jumps) / len(jumps) if jumps else None,
            }
            rows_out.append(row)
        b = e

    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        'purpose': 'Use AprilTag only as the distance reference; evaluate FoundationPose continuity/stability versus distance.',
        'distance_range_m': [dmin, dmax],
        'distance_bin_width_m': args.bin_m,
        'tag_detected_common_frames': len(samples),
        'quasi_stationary_definition': {
            'tag_translation_step_mm_max': args.stationary_trans_mm,
            'tag_rotation_step_deg_max': args.stationary_rot_deg,
        },
        'large_fp_jump_definition': {
            'fp_translation_step_mm_gt': args.fp_jump_mm,
            'fp_symmetry_aware_rotation_step_deg_gt': args.fp_jump_rot_deg,
        },
        'important_limitation': 'The recording contains hand motion. All-frame FP step metrics contain both true object motion and tracker jitter. Quasi-stationary metrics are the better stability indicator, but the cleanest final range test is still a short stationary hold at each distance.',
        'bins': rows_out,
    }
    out_json.write_text(json.dumps(summary, indent=2))

    print('OFFLINE FOUNDATIONPOSE RANGE ANALYSIS')
    print(f'AprilTag distance range: {dmin:.3f} to {dmax:.3f} m')
    print('Tag is used as the distance reference, not as perfect 6D ground truth.')
    print('')
    print('bin(m)      frames  stat_pairs  FP stat step mm med/p95  FP stat rot deg med/p95  jump%')
    for r in rows_out:
        print(
            f"{r['distance_bin_m']:11s} {r['frames']:6d} {r['quasi_stationary_pairs']:11d}  "
            f"{fmt(r['fp_step_mm_median_stationary']):>6s}/{fmt(r['fp_step_mm_p95_stationary']):<6s}  "
            f"{fmt(r['fp_step_rot_deg_median_stationary']):>6s}/{fmt(r['fp_step_rot_deg_p95_stationary']):<6s}  "
            f"{fmt(r['large_jump_rate_pct_all']):>6s}"
        )

    print('')
    print('Interpretation: use the quasi-stationary p95 columns to judge where FP starts becoming unstable.')
    print('All-frame jump rate is only a continuity warning because the box was moved by hand.')
    print('saved:', out_csv)
    print('saved:', out_json)


if __name__ == '__main__':
    main()
