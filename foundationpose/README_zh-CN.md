# FoundationPose RGB-D 箱体位姿跟踪

[English](readme.md) | **简体中文**

本仓库基于 [NVIDIA FoundationPose](https://github.com/NVlabs/FoundationPose)，用于**已知包装箱的 6D 位姿估计与连续 RGB-D 跟踪**。

本仓库只聚焦于**视觉模块**：RGB-D 数据采集、深度到彩色图像对齐、一次性物体注册、FoundationPose 连续跟踪、运行速度测试、离线回放以及跟踪稳定性分析。

```text
RGB + 对齐后的深度 + 相机内参 + 箱体网格
                         ↓
                 一次性物体掩码
                         ↓
              FoundationPose register()
                         ↓
               连续执行 track_one()
                         ↓
                相机坐标系下的 6D 箱体位姿
```

## 演示

![FoundationPose RGB-D 箱体跟踪](assets/demo/foundationpose_demo.gif)

**完整连续跟踪序列：** [观看完整视频](assets/demo/foundationpose_full.mp4)

上面的动图是录制 RGB-D 跟踪序列中的 10 秒片段。画面中包含 FoundationPose 的箱体框与坐标轴，以及跟踪 FPS、跟踪耗时、帧编号和估计距离。

---

# 环境配置

下面给出当前实现所使用的可复现环境配置。

### 已测试的软件栈

- Ubuntu Linux
- Python **3.11**
- CUDA Toolkit **12.8**
- PyTorch **2.7.1 + cu128**
- torchvision **0.22.1 + cu128**
- torchaudio **2.7.1 + cu128**
- 从源码编译 PyTorch3D
- 从源码编译 NVDiffRast

需要较新的 NVIDIA GPU。当前高帧率基准测试使用 RTX 5080 完成。

## 1. 克隆仓库

```bash
git clone https://github.com/Chris-airobot/FoundationPose.git
cd FoundationPose
```

## 2. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate foundationpose
```

`environment.yml` 会安装 Python 3.11，以及编译 CUDA 扩展所需的编译器和 CMake 依赖。

## 3. 安装 PyTorch CUDA 12.8

```bash
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

检查 CUDA 是否可用：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda:', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

## 4. 配置 CUDA Toolkit

如果 CUDA 12.8 安装在 `/usr/local/cuda-12.8`：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export C_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
```

如果 CUDA 安装在其他位置，请相应修改 `CUDA_HOME`。

## 5. 编译 / 安装 GPU 依赖

```bash
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git"

python -m pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast.git"

python -m pip install -r requirements.txt
bash build_all_conda.sh
```

## 6. 验证环境

```bash
python check_env.py
```

## 7. 下载 FoundationPose 权重

从上游项目下载官方预训练权重，并按照以下结构放到 `weights/` 目录：

```text
weights/
├── 2023-10-28-18-33-37/    # refiner
│   ├── model_best.pth
│   └── config.yml
└── 2024-01-11-20-02-45/    # scorer
    ├── model_best.pth
    └── config.yml
```

NVIDIA FoundationPose 官方权重链接：

https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing

更多安装说明见 [`ENV_SETUP.md`](ENV_SETUP.md)。

---

# 箱体模型

当前物体近似为一个长方体包装箱：

```text
Length: 0.40 m
Width:  0.30 m
Height: 0.30 m
```

网格文件为：

```text
box.obj
```

可以重新生成：

```bash
python makee_box.py
```

模型原点位于箱体中心，网格单位为米。

由于 CAD / 网格模型已知，本仓库使用**基于模型的 FoundationPose 流程**，不需要针对该物体单独重新训练网络。

---

# 输入格式

FoundationPose 需要：

- RGB 图像
- 深度图像
- 相机内参矩阵 `K`
- 物体网格
- 用于初始注册的一张物体掩码

典型录制序列结构如下：

```text
sequence/
├── rgb/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── depth/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── masks/
│   └── 000000.png
└── cam_K.txt
```

深度以 16 位毫米值保存，传入 FoundationPose 前会转换为米。

正确的**深度到彩色图像对齐**非常重要。如果 RGB 与深度使用不一致的图像几何关系，位姿估计质量会明显下降。

---

# 基础模型测试

对于已经按照 FoundationPose 格式保存的 RGB-D 序列：

```bash
conda activate foundationpose

python run_demo.py \
  --mesh_file box.obj \
  --test_scene_dir demo_data/box_test \
  --debug 2 \
  --debug_dir debug_box_test
```

FoundationPose 会在第一帧执行 `register()`，随后对后续帧自动执行 `track_one()`。

---

# RealSense RGB-D 采集

仓库提供独立的 RealSense 录制脚本：

```bash
python -m pip install pyrealsense2
python capture_realsense_box.py --list-devices
```

录制一个序列：

```bash
python capture_realsense_box.py \
  --serial REPLACE_WITH_SERIAL \
  --output-dir demo_data/box_test
```

该录制脚本会：

- 将深度对齐到彩色图像；
- 保存 RGB 帧；
- 以 16 位毫米值保存深度；
- 将相机内参保存到 `cam_K.txt`；
- 交互式创建第一帧掩码。

---

# 实时跟踪

主要实时跟踪脚本为：

```text
g1/scripts/live_foundationpose_box.py
```

该脚本通过 ZMQ 接收 640 × 480 RGB-D 数据流，完成一次注册后持续跟踪箱体。

```bash
conda activate foundationpose
python -u g1/scripts/live_foundationpose_box.py
```

控制键：

```text
q  退出
r  重新绘制物体掩码并重新注册
```

最新估计得到的 4×4 位姿矩阵保存在：

```text
g1/results/live_foundationpose/latest_pose.txt
```

该位姿表示物体在**相机坐标系**中的变换。

实时循环使用：

- `TRACK_ITERS = 1`；
- ZMQ `CONFLATE=1`，丢弃旧帧以避免延迟累积；
- 降低可视化频率；
- 减少磁盘写入。

终端会输出：

```text
camera-arrival FPS=... | pose-output FPS=... | track=... ms
```

---

# 运行速度基准测试

核心基准测试脚本：

```bash
python -u g1/scripts/benchmark_foundationpose_core.py
```

该脚本会先将 RGB-D 帧预加载到内存，执行一次注册，对跟踪器进行预热，然后在不包含常规可视化和磁盘 I/O 开销的情况下反复测量 `track_one()`。

### 实测结果

| 指标 | 结果 |
|---|---:|
| 核心跟踪吞吐率 | **104.7 FPS** |
| 平均跟踪时间 | **9.55 ms/frame** |
| p95 跟踪时间 | **12.51 ms/frame** |

这些数字测量的是 FoundationPose 跟踪核心本身。完整实时系统的吞吐率可能更低，因为还会受到相机采集、对齐、传输、解码和可视化等额外开销影响。

---

# 距离与跟踪稳定性结果

使用专门录制的一段 RGB-D 数据，在箱体与相机距离发生变化时评估 FoundationPose 的跟踪稳定性。

### 录制数据集

| 项目 | 结果 |
|---|---:|
| 帧数 | **2692** |
| 采集帧率 | **29.91 FPS** |
| RGB / 深度可读率 | **100%** |
| RGB / 深度尺寸匹配率 | **100%** |
| 平均有效深度比例 | **97.7%** |
| AprilTag 检测覆盖率 | **91.3%** |
| 测试距离范围 | **1.124–1.902 m** |

AprilTag 主要作为**距离参考**使用，并未被视为 FoundationPose 的完美 6D 真值。

分析使用 FoundationPose 相邻帧之间的平移变化以及考虑箱体对称性的旋转变化。对于所有有数据的距离区间中的准静止帧对：

| 跟踪稳定性指标 | 结果 |
|---|---:|
| p95 平移步长 | **≤ ~13 mm** |
| p95 对称性旋转步长 | **≤ ~4.4°** |
| 大跳变比例 | **0%** |

### 结论

在大约 **1.1–1.9 m** 的完整测试范围内，FoundationPose 始终保持稳定，没有观察到明显的突然跟踪退化。

较为保守的实际解释是：

- **1.1–1.9 m：** 在本次录制实验中验证过的可用跟踪范围；
- **≤1.8 m：** 现有样本能够较充分支持；
- **>1.9 m：** 本次基准测试尚未验证。

1.8–1.9 m 区间的准静止样本较少（23 对），因此 `≤1.8 m` 是更保守的结论。

### 该结果的重要限制

序列首先进行**一次成功的 FoundationPose 注册**，之后在距离变化过程中持续执行 `track_one()`。因此，该实验验证的是**跟踪稳定性随距离的变化情况**，而不是在每个距离重新执行一次注册时的成功概率。

由于实验中箱体由手持移动，准静止相邻帧之间的变化仍可能包含少量真实物体运动。因此，这些数值更适合作为一种上界式稳定性指标，而不是纯粹的估计器噪声。

分析脚本包括：

```text
g1/scripts/range_benchmark.py
g1/scripts/offline_fp_range_analysis.py
g1/scripts/compare_offline_fp_apriltag.py
g1/scripts/apriltag_gt_benchmark.py
```

---

# 离线录制与回放

RGB-D 数据录制完成后，可以完全离线评估整个视觉流程。

常用脚本：

| 脚本 | 用途 |
|---|---|
| `g1/scripts/record_offline_bundle.py` | 录制可重复使用的 RGB-D 数据包 |
| `g1/scripts/verify_offline_bundle.py` | 验证 RGB / 深度数据并提取 AprilTag 测量结果 |
| `g1/scripts/foundationpose_offline_bundle.py` | 在录制数据包上运行 FoundationPose |
| `g1/scripts/replay_foundationpose_benchmark.py` | 回放 / 基准测试录制序列 |
| `g1/scripts/compare_offline_fp_apriltag.py` | 使用 AprilTag 参考几何进行可视化一致性检查 |
| `g1/scripts/offline_fp_range_analysis.py` | 分析不同距离下的跟踪稳定性 |

这样无需重复采集相同数据，就可以反复执行跟踪、性能和距离分析实验。

---

# 项目结构

```text
FoundationPose/
├── box.obj
├── makee_box.py
├── estimater.py
├── run_demo.py
├── capture_realsense_box.py
├── check_env.py
├── environment.yml
├── requirements.txt
├── ENV_SETUP.md
│
└── g1/
    ├── camera_server/              # RGB-D 相机集成 / 对齐
    ├── data/                       # 小型配置 / 示例数据
    └── scripts/
        ├── live_foundationpose_box.py
        ├── make_first_mask.py
        ├── record_offline_bundle.py
        ├── verify_offline_bundle.py
        ├── foundationpose_offline_bundle.py
        ├── replay_foundationpose_benchmark.py
        ├── benchmark_foundationpose_core.py
        ├── range_benchmark.py
        ├── offline_fp_range_analysis.py
        ├── apriltag_gt_benchmark.py
        └── compare_offline_fp_apriltag.py
```

大型 RGB-D 录制数据、生成的可视化结果、基准测试输出和日志不应加入常规 Git 提交。

---

# 当前限制

当前主要剩余的感知限制是初始化：

> **第一张物体掩码需要手动提供。**

完成注册后，FoundationPose 可以持续跟踪，不需要每一帧都重新提供分割掩码。如果以后需要全自动初始化，可以进一步加入目标检测器或分割模型。

---

# 上游 FoundationPose

本仓库基于 NVIDIA Research 官方实现：

- Repository: https://github.com/NVlabs/FoundationPose
- Project page: https://nvlabs.github.io/FoundationPose/
- Paper: *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects*, CVPR 2024 Highlight

原始预训练权重、公开数据集使用说明、model-free 流程以及完整方法说明，请参考上游项目。

## 引用

```bibtex
@InProceedings{foundationposewen2024,
  author    = {Bowen Wen and Wei Yang and Jan Kautz and Stan Birchfield},
  title     = {FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects},
  booktitle = {CVPR},
  year      = {2024}
}
```

# License

上游 FoundationPose 代码和数据按照 NVIDIA Source Code License 发布。具体许可信息请查看 `LICENSE` 和 NVIDIA 原始仓库。
