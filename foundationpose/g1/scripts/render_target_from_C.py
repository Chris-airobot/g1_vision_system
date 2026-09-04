import csv
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from estimater import draw_posed_3d_box, draw_xyz_axis

ROOT = Path('g1/data/current_task')
POSES = ROOT / 'track_C' / 'poses.csv'
RGB_DIR = ROOT / 'rgb'
OUT_DIR = ROOT / 'target_from_C'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Neighboring same-size box center offset established previously.
OFFSET_M = 0.30

# Save enough frames to inspect, especially the useful joint-view/final region.
KEY_FRAMES = set([
    245, 250, 270, 300, 320, 331, 337, 350,
    380, 400, 420, 436, 450, 470, 500, 520, 540, 560, 580
])

meta = json.loads((ROOT / 'intrinsics.json').read_text())
K = np.array([
    [float(meta['fx']), 0.0, float(meta['cx'])],
    [0.0, float(meta['fy']), float(meta['cy'])],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

mesh = trimesh.load('box.obj', force='mesh')
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
center_tf = np.linalg.inv(to_origin)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

# We use a local-axis offset from C so the desired neighboring target follows C's
# orientation instead of depending on the moving handheld camera axes.
# First test both +/- along C local Y. One of these should fall into the empty slot.
T_C_B_plus = np.eye(4)
T_C_B_plus[1, 3] = +OFFSET_M

T_C_B_minus = np.eye(4)
T_C_B_minus[1, 3] = -OFFSET_M

rows = list(csv.DictReader(POSES.open()))
print('pose rows:', len(rows))
print('mesh extents:', extents)
print('candidate offset:', OFFSET_M, 'm along C local Y')

saved = 0
for row in rows:
    frame = int(row['frame'])
    if frame not in KEY_FRAMES and frame % 20 != 0:
        continue

    img_path = RGB_DIR / f'{frame:06d}.png'
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    T_cam_C = np.array([
        [float(row[f't{r}{c}']) for c in range(4)]
        for r in range(4)
    ], dtype=np.float64)

    T_cam_B_plus = T_cam_C @ T_C_B_plus
    T_cam_B_minus = T_cam_C @ T_C_B_minus

    C_center = T_cam_C @ center_tf
    Bp_center = T_cam_B_plus @ center_tf
    Bm_center = T_cam_B_minus @ center_tf

    vis = rgb.copy()

    # Reference C: normal FoundationPose cuboid + axes.
    vis = draw_posed_3d_box(K, img=vis, ob_in_cam=C_center, bbox=bbox)
    vis = draw_xyz_axis(
        vis,
        ob_in_cam=C_center,
        scale=0.10,
        K=K,
        thickness=2,
        transparency=0,
        is_input_rgb=True,
    )

    # Candidate B+: thicker cuboid.
    vis = draw_posed_3d_box(K, img=vis, ob_in_cam=Bp_center, bbox=bbox)

    # Candidate B-: cuboid as well.
    vis = draw_posed_3d_box(K, img=vis, ob_in_cam=Bm_center, bbox=bbox)

    out = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.putText(out, f'C + B candidates | frame {frame}', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)

    # Label projected target centers so the two candidates are distinguishable.
    for name, cp, color in [
        ('B+', Bp_center, (0,0,255)),
        ('B-', Bm_center, (255,0,255)),
    ]:
        p = cp[:3,3]
        if p[2] > 0:
            uv = K @ p
            uv = uv[:2] / uv[2]
            u,v = int(round(uv[0])), int(round(uv[1]))
            if 0 <= u < out.shape[1] and 0 <= v < out.shape[0]:
                cv2.circle(out, (u,v), 7, color, -1)
                cv2.putText(out, name, (u+8,v-8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2, cv2.LINE_AA)

    cv2.imwrite(str(OUT_DIR / f'{frame:06d}.jpg'), out)
    saved += 1

print('saved visualizations:', saved)
print('output:', OUT_DIR)
