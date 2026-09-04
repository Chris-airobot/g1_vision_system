"""极简 OBJ 读取，只取顶点和三角面 —— 够可视化用，不引第三方依赖（除 numpy）。

不处理材质、法线、纹理。多边形面会做扇形三角化。
"""

from __future__ import annotations

from pathlib import Path

#: 随工程附带的 Ultimate Tracker CAD 模型（``ultimate_tracker_3d/``）与 tracker
#: **位姿本体系**之间的轴置换，实测目视对齐 + 物理自洽性检验得到。
#:
#: 规格含义：CAD 的 X / Y / Z 轴分别指向本体系的哪个轴 ::
#:
#:     CAD X（长 79 mm） -> -本体 X
#:     CAD Y（长 59 mm） -> -本体 Z
#:     CAD Z（厚 27 mm） -> -本体 Y
#:
#: 独立验证：设备平躺时，CAD 的厚度方向（27 mm 那一维）在世界系里与竖直方向
#: 只差 1.1° —— 薄的一维朝上，正是平躺应有的样子。之前几个候选都过不了这一关。
#:
#: 只影响模型显示，对位姿数据本身无影响。但若要用 CAD 里的安装孔位/尺寸推算
#: tracker 相对机器人 base_link 的外参，就需要它。
MESH_AXES_CAD_TO_RAW_BODY = "-X,-Z,-Y"

#: 对应的四元数 (x, y, z, w)，等价绕 (0,-1,1)/√2 转 180°
MESH_QUAT_RAW_BODY_FROM_CAD = (0.0, -0.7071067811865476, 0.7071067811865476, 0.0)

#: 选定的机体系轴重映射（相对设备原始机体系）。
#: 目的是让 **X 轴沿设备厚度方向** ::
#:
#:     新 X = +旧 Y = -CAD Z   厚 27 mm
#:     新 Y = -旧 X = +CAD X   长 79 mm
#:     新 Z = +旧 Z = -CAD Y      59 mm
BODY_AXES_REMAP = "Y,-X,Z"

#: CAD 轴 -> **重映射后**的机体轴。上面两项的合成。挂在已重映射的机体系下时直接用它
#: （g1_tracker_system 的 URDF 网格就是这样生成的）。``vvu-viz`` 例外：它会把 ``--body-axes`` 的换轴补偿回去，
#: 所以它的 ``--mesh-preset`` 要给 :data:`MESH_AXES_CAD_TO_RAW_BODY`。
MESH_AXES_CAD_TO_BODY = "Y,-Z,-X"

#: 对应四元数 (x, y, z, w)
MESH_QUAT_BODY_FROM_CAD = (-0.5, -0.5, 0.5, 0.5)

MESH_SCALE = 0.001


def load_obj(path: str | Path, scale: float = 1.0):
    """读 OBJ，返回 ``(vertices (N,3) float32, faces (M,3) uint32)``。

    :param scale: 顶点缩放。CAD 常见单位是 mm，转米传 ``0.001``。
    """
    try:
        import numpy as np
    except ImportError as exc:                        # pragma: no cover
        raise ImportError('load_obj 需要 numpy：pip install -e ".[analysis]"') from exc

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line:
                continue
            tag = line[:2]
            if tag == "v ":
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif tag == "f ":
                # "f v/vt/vn v/vt/vn ..." -> 只要第一个索引；OBJ 索引从 1 开始
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):      # 扇形三角化
                    faces.append((idx[0], idx[k], idx[k + 1]))

    v = np.asarray(verts, dtype=np.float32)
    if scale != 1.0:
        v = v * float(scale)
    return v, np.asarray(faces, dtype=np.uint32)


def bounds(vertices):
    """返回 ``(min, max, size)``，用来确认单位和原点位置。"""
    import numpy as np
    v = np.asarray(vertices)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return lo, hi, hi - lo
