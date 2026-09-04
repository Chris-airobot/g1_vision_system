import csv, argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Derive static T_robot_camera from one independently measured T_robot_marker and the simultaneous vision T_camera_marker.')
    p.add_argument('--bundle', required=True)
    p.add_argument('--robot-marker', required=True, help='4x4 txt: independently measured T_robot_marker for the selected reference frame.')
    p.add_argument('--frame', type=int, default=-1, help='Reference frame index. Default: first detected AprilTag frame.')
    return p.parse_args()


def row_to_T(row):
    T = np.eye(4)
    T[:3,3] = [float(row['tx_m']), float(row['ty_m']), float(row['tz_m'])]
    for i in range(3):
        for j in range(3):
            T[i,j] = float(row[f'r{i}{j}'])
    return T


def main():
    args = parse_args()
    bundle = Path(args.bundle)
    pose_csv = bundle / 'apriltag_poses.csv'
    if not pose_csv.exists():
        raise RuntimeError('apriltag_poses.csv missing. Run verify_offline_bundle.py first.')

    rows = list(csv.DictReader(pose_csv.open()))
    detected = [r for r in rows if str(r['detected']).lower() == 'true']
    if not detected:
        raise RuntimeError('No detected AprilTag pose in apriltag_poses.csv')

    if args.frame >= 0:
        matches = [r for r in detected if int(r['frame']) == args.frame]
        if not matches:
            raise RuntimeError(f'Frame {args.frame} has no AprilTag detection')
        row = matches[0]
    else:
        row = detected[0]

    T_robot_marker = np.loadtxt(args.robot_marker).reshape(4,4)
    T_camera_marker = row_to_T(row)

    # T_robot_marker = T_robot_camera @ T_camera_marker
    # => T_robot_camera = T_robot_marker @ inv(T_camera_marker)
    T_robot_camera = T_robot_marker @ np.linalg.inv(T_camera_marker)
    out = bundle / 'T_robot_camera.txt'
    np.savetxt(out, T_robot_camera)

    print('reference frame:', row['frame'])
    print('T_robot_camera =')
    print(T_robot_camera)
    print('saved:', out)
    print('IMPORTANT: valid only if T_robot_marker corresponds to this exact recorded frame and frame conventions are consistent.')


if __name__ == '__main__':
    main()
