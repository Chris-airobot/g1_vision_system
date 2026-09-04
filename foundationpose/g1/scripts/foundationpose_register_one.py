import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from estimater import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rgb', required=True)
    p.add_argument('--depth', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--intrinsics', required=True)
    p.add_argument('--mesh', default='box.obj')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--register-iters', type=int, default=5)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # RGB
    bgr = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f'Cannot read RGB: {args.rgb}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Raw RealSense depth -> metres
    dep_raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if dep_raw is None:
        raise RuntimeError(f'Cannot read depth: {args.depth}')

    meta = json.loads(Path(args.intrinsics).read_text())
    scale = float(meta['depth_scale_m_per_unit'])
    depth = dep_raw.astype(np.float32) * scale
    depth[(depth < 0.001) | (depth > 10.0)] = 0

    # Binary initialization mask
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f'Cannot read mask: {args.mask}')
    mask = mask > 0

    K = np.array([
        [float(meta['fx']), 0.0, float(meta['cx'])],
        [0.0, float(meta['fy']), float(meta['cy'])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    mesh = trimesh.load(args.mesh, force='mesh')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    center_tf = np.linalg.inv(to_origin)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    set_logging_format()
    set_seed(0)

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

    pose = est.register(
        K=K,
        rgb=rgb,
        depth=depth,
        ob_mask=mask,
        iteration=args.register_iters,
    )

    cp = pose @ center_tf

    np.savetxt(out / 'pose.txt', pose)
    np.savetxt(out / 'center_pose.txt', cp)
    np.savetxt(out / 'cam_K.txt', K)

    vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=cp, bbox=bbox)
    vis = draw_xyz_axis(
        vis,
        ob_in_cam=cp,
        scale=0.1,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / 'pose_vis.png'), vis_bgr)

    center = cp[:3, 3]
    print('FOUNDATIONPOSE REGISTER: PASS')
    print('output:', out)
    print('mesh extents:', extents)
    print('mask pixels:', int(mask.sum()))
    print('center xyz camera m:', center.tolist())
    print('distance camera m:', float(np.linalg.norm(center)))
    print('pose saved:', out / 'pose.txt')
    print('visualization saved:', out / 'pose_vis.png')


if __name__ == '__main__':
    main()
