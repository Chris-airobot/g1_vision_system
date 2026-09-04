import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path('g1/data/current_task')
RGB_DIR = ROOT / 'rgb'
A_CSV = ROOT / 'track_A' / 'poses.csv'
C_CSV = ROOT / 'track_C' / 'poses.csv'
OUT = ROOT / 'A_C_target_full.mp4'

TARGET_OFFSET_M = 0.30
TARGET_SIGN = +1.0   # change to -1.0 only if target is on wrong side
FPS = 30.0

meta = json.loads((ROOT / 'intrinsics.json').read_text())
K = np.array([
    [float(meta['fx']), 0.0, float(meta['cx'])],
    [0.0, float(meta['fy']), float(meta['cy'])],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

mesh = trimesh.load('box.obj', force='mesh')
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
center_tf = np.linalg.inv(to_origin)
bbox = np.stack([-extents/2, extents/2], axis=0)

print('mesh extents:', extents)
print('target offset:', TARGET_SIGN * TARGET_OFFSET_M, 'm along C local Y')


def load_poses(path):
    poses = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            frame = int(row['frame'])
            T = np.array([
                [float(row[f't{r}{c}']) for c in range(4)]
                for r in range(4)
            ], dtype=np.float64)
            poses[frame] = T
    return poses


A = load_poses(A_CSV)
C = load_poses(C_CSV)
print('A poses:', len(A), 'range:', min(A), '->', max(A))
print('C poses:', len(C), 'range:', min(C), '->', max(C))

T_C_B = np.eye(4, dtype=np.float64)
T_C_B[1, 3] = TARGET_SIGN * TARGET_OFFSET_M


def project(T, point):
    p = T[:3,:3] @ point + T[:3,3]
    if p[2] <= 1e-6:
        return None
    uv = K @ p
    uv = uv[:2] / uv[2]
    return tuple(np.round(uv).astype(int))


def draw_box(img, T_cam_obj, color, thickness=3):
    # FoundationPose pose is converted using the same oriented-bounds center
    # transform used in the successful one-frame/C-target visualizations.
    T = T_cam_obj @ center_tf

    lo, hi = bbox
    corners = np.array([
        [lo[0],lo[1],lo[2]], [hi[0],lo[1],lo[2]],
        [hi[0],hi[1],lo[2]], [lo[0],hi[1],lo[2]],
        [lo[0],lo[1],hi[2]], [hi[0],lo[1],hi[2]],
        [hi[0],hi[1],hi[2]], [lo[0],hi[1],hi[2]],
    ], dtype=np.float64)

    pts = [project(T,p) for p in corners]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]

    for i,j in edges:
        if pts[i] is not None and pts[j] is not None:
            cv2.line(img, pts[i], pts[j], color, thickness, cv2.LINE_AA)

    center = project(T, np.zeros(3))
    if center is not None:
        cv2.circle(img, center, 5, color, -1, cv2.LINE_AA)
    return center


rgb_files = sorted(RGB_DIR.glob('*.png'))
assert rgb_files, 'No RGB frames found'

first = cv2.imread(str(rgb_files[0]))
assert first is not None
h,w = first.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (w,h))
assert writer.isOpened(), 'Could not open VideoWriter'

for idx,p in enumerate(rgb_files):
    frame = int(p.stem)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        continue

    # OpenCV BGR colors.
    A_COLOR = (0,0,255)      # red
    C_COLOR = (0,255,0)      # green
    B_COLOR = (255,0,0)      # blue

    if frame in A:
        ca = draw_box(img, A[frame], A_COLOR, 3)
        if ca is not None:
            cv2.putText(img,'A carried',(ca[0]+8,ca[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,A_COLOR,2,cv2.LINE_AA)

    if frame in C:
        cc = draw_box(img, C[frame], C_COLOR, 3)
        if cc is not None:
            cv2.putText(img,'C reference',(cc[0]+8,cc[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,C_COLOR,2,cv2.LINE_AA)

        T_cam_B = C[frame] @ T_C_B
        cb = draw_box(img, T_cam_B, B_COLOR, 4)
        if cb is not None:
            cv2.putText(img,'B TARGET',(cb[0]+8,cb[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.60,B_COLOR,2,cv2.LINE_AA)

    # Fixed legend.
    cv2.rectangle(img,(8,8),(215,92),(0,0,0),-1)
    cv2.putText(img,'RED   A carried',(18,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,A_COLOR,2)
    cv2.putText(img,'GREEN C reference',(18,54),cv2.FONT_HERSHEY_SIMPLEX,0.5,C_COLOR,2)
    cv2.putText(img,'BLUE  B target',(18,78),cv2.FONT_HERSHEY_SIMPLEX,0.5,B_COLOR,2)
    cv2.putText(img,f'frame {frame}',(w-145,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

    writer.write(img)

    if idx % 100 == 0:
        print('rendered', idx, '/', len(rgb_files), 'frame', frame)

writer.release()
print('VIDEO SAVED:', OUT)
