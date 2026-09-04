#!/usr/bin/env python3
"""把 Ultimate Tracker 的 CAD（OBJ，毫米）转成 URDF 用的二进制 STL（米）。

输出网格直接表达在 ``vive_tracker_link`` 的链接系里，URDF 的 visual origin 用单位阵即可。

链接系 = viva_vive_ultimate 的「重映射 tracker 机体系 T」（``meshio.MESH_AXES_CAD_TO_BODY``）::

    body X = -CAD Z   厚度方向 27 mm，从摄像头面指向安装面（装上机器人后指向机器人）
    body Y = +CAD X   长边 79 mm
    body Z = -CAD Y   短边 59 mm

原点 = 安装面（CAD z_min 那块平板）的中心。整个 tracker 位于 body x ∈ [-0.0273, 0]。

用法::

    python make_tracker_mesh.py            # 默认路径
    python make_tracker_mesh.py --obj ... --out ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from viva_vive_ultimate.frames import parse_axis_remap
from viva_vive_ultimate.meshio import MESH_AXES_CAD_TO_BODY, MESH_SCALE, load_obj

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_config as cfg

DEFAULT_OBJ = (cfg.PROJECT_DIR.parent / "ultimate_tracker_3d"
               / "Ultimate Tracker 3D" / "Ultimate Tracker_1115.obj")
DEFAULT_OUT = cfg.MODEL_DIR / "meshes" / "vive_ultimate_tracker.STL"

#: 安装面判定：取 CAD z 最低 0.5 mm 内的顶点当安装平板
MOUNT_BAND_MM = 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--obj", type=Path, default=DEFAULT_OBJ)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    import trimesh

    v_cad, faces = load_obj(args.obj)          # 毫米，CAD 系
    v_cad = np.asarray(v_cad, dtype=np.float64)

    # 安装面：z_min 平板的中心
    z_min = float(v_cad[:, 2].min())
    band = v_cad[v_cad[:, 2] < z_min + MOUNT_BAND_MM]
    mount_cad = np.array([
        0.5 * (band[:, 0].min() + band[:, 0].max()),
        0.5 * (band[:, 1].min() + band[:, 1].max()),
        z_min,
    ])
    print(f"CAD bbox min {v_cad.min(0)}  max {v_cad.max(0)}  (mm)")
    print(f"安装面平板: z={z_min:.2f} mm, 平板范围 "
          f"{np.ptp(band[:, 0]):.1f} x {np.ptp(band[:, 1]):.1f} mm, 中心 {mount_cad[:2]}")

    # CAD -> 机体系（旋转），再平移使安装面中心落在原点，再 mm -> m
    R = parse_axis_remap(MESH_AXES_CAD_TO_BODY)      # v_body = R @ v_cad，自带 det=+1 检查
    v_body = (R @ (v_cad - mount_cad).T).T * MESH_SCALE

    mesh = trimesh.Trimesh(vertices=v_body, faces=np.asarray(faces), process=True)
    lo, hi = mesh.bounds
    print(f"链接系 bbox min {lo}  max {hi}  (m)")
    print(f"三角面 {len(mesh.faces)}，顶点 {len(mesh.vertices)}")

    # 自检：本体应完全在安装面后方（x<=0），厚 27.3、长 79、宽 59
    assert hi[0] <= 1e-6 and -0.0280 < lo[0] < -0.0265, "厚度方向不对"
    assert 0.078 < hi[1] - lo[1] < 0.080, "Y 应是 79 mm 长边"
    assert 0.058 < hi[2] - lo[2] < 0.060, "Z 应是 59 mm 短边"
    assert abs(hi[1] + lo[1]) < 0.002 and abs(hi[2] + lo[2]) < 0.002, "安装面中心没对到原点"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out, file_type="stl")          # trimesh 默认二进制 STL
    print(f"写入 {args.out}  ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
