import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from estimater import *

REGISTER_ITERS = 5
TRACK_ITERS = 1
DEPTH_MAX_M = 10.0


def load_intrinsics(path):
    meta = json.loads(Path(path).read_text())
    K = np.array([
        [float(meta['fx']), 0.0, float(meta['cx'])],
        [0.0, float(meta['fy']), float(meta['cy'])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    scale = float(meta['depth_scale_m_per_unit'])
    return K, scale


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f'cannot read RGB: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(path, scale):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f'cannot read depth: {path}')
    dep = raw.astype(np.float32) * scale
    dep[(dep < 0.001) | (dep > DEPTH_MAX_M)] = 0
    return dep


def make_estimator(mesh, outdir):
    return FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=str(outdir),
        debug=0,
        glctx=dr.RasterizeCudaContext(),
    )


def run_object(label, init_frame, mask_path, frames, depth_frames, K, scale, mesh, root):
    out = root / f'track_{label}'
    vis_dir = out / 'vis'
    out.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    center_tf = np.linalg.inv(to_origin)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    est = make_estimator(mesh, out)

    rgb0 = load_rgb(frames[init_frame])
    dep0 = load_depth(depth_frames[init_frame], scale)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f'cannot read mask: {mask_path}')
    mask = mask > 0

    print(f'[{label}] register at frame {init_frame}')
    t0 = time.perf_counter()
    pose = est.register(K=K, rgb=rgb0, depth=dep0, ob_mask=mask, iteration=REGISTER_ITERS)
    print(f'[{label}] registration ms={(time.perf_counter()-t0)*1000.0:.1f}')

    csv_path = out / 'poses.csv'
    with csv_path.open('w', newline='') as f:
        fields = ['frame','track_ms','center_x','center_y','center_z','distance'] + [f't{r}{c}' for r in range(4) for c in range(4)]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for idx in range(init_frame, len(frames)):
            if idx == init_frame:
                rgb = rgb0
                track_ms = 0.0
            else:
                rgb = load_rgb(frames[idx])
                dep = load_depth(depth_frames[idx], scale)
                q0 = time.perf_counter()
                pose = est.track_one(rgb=rgb, depth=dep, K=K, iteration=TRACK_ITERS)
                track_ms = (time.perf_counter()-q0)*1000.0

            cp = pose @ center_tf
            c = cp[:3,3]
            row = {
                'frame': idx,
                'track_ms': track_ms,
                'center_x': float(c[0]),
                'center_y': float(c[1]),
                'center_z': float(c[2]),
                'distance': float(np.linalg.norm(c)),
            }
            for r in range(4):
                for cc in range(4):
                    row[f't{r}{cc}'] = float(pose[r,cc])
            w.writerow(row)

            # Save diagnostic overlays every 10 frames plus key regions.
            save_vis = (idx % 10 == 0) or idx in [195,245,271,278,286,300,320,331,337,350,390,406,436,450,len(frames)-1]
            if save_vis:
                vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=cp, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=cp, scale=0.1, K=K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, f'{label} frame {idx}', (10,28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0,255,0), 2, cv2.LINE_AA)
                cv2.imwrite(str(vis_dir / f'{idx:06d}.jpg'), bgr)

            if idx > init_frame and idx % 50 == 0:
                print(f'[{label}] frame={idx} center={c.tolist()} track_ms={track_ms:.1f}')

    np.savetxt(out / 'latest_pose.txt', pose)
    print(f'[{label}] DONE -> {csv_path}')


def main():
    set_logging_format()
    set_seed(0)

    root = Path('g1/data/current_task')
    rgb = sorted((root/'rgb').glob('*.png'))
    depth = sorted((root/'depth').glob('*.png'))
    assert rgb and len(rgb) == len(depth)

    K, scale = load_intrinsics(root/'intrinsics.json')
    mesh = trimesh.load('box.obj', force='mesh')

    print('frames:', len(rgb))
    print('K:\n', K)
    print('depth scale:', scale)
    print('mesh extents:', mesh.extents)

    run_object('A', 195, root/'mask_A_init.png', rgb, depth, K, scale, mesh, root)
    run_object('C', 245, root/'mask_C_init.png', rgb, depth, K, scale, mesh, root)

    print('TRACKING RUN COMPLETE')

if __name__ == '__main__':
    main()
