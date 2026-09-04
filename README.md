# G1 Box Tracking

Unified development repo for Samsung Unitree G1 box perception/tracking.

## Components

- `vive/`
  - VIVE Ultimate tracker interface
  - G1 root/common-world tracking
  - ChArUco calibration
  - G1 camera/root visualization

- `foundationpose/`
  - FoundationPose source and G1 box scripts
  - 40 x 30 x 30 cm `box.obj`
  - heavy weights/data are intentionally excluded

- `integration/`
  - unified G1 + VIVE + dual-camera + FoundationPose system

## Common frames

- `K`: common/world frame
- `E`: external D435i optical frame
- `B`: G1 pelvis/root
- `C`: G1 onboard camera
- `T`: VIVE tracker
- `O`: box

External camera box:
`K_T_O_ext = K_T_E @ E_T_O`

G1 camera box:
`K_T_O_g1 = K_T_C @ C_T_O`

where:
`K_T_C = K_T_B @ B_T_C(q)`

The two box estimates remain independent initially and are compared in K.

## Unified integration runtime

`integration/g1_unified_vision.py` is the first combined runtime. It retains
the existing VIVE body-axis remap, rev1.0 URDF waist FK, 30-frame automatic
VIVE/vision alignment, and tracker-to-root calibration. Camera acquisition and
FoundationPose inference run outside the visualization loop, so a slow or
missing camera/inference result does not stall the 3D UI.

The external and onboard streams each have their own `FoundationPoseWorker`.
Each worker constructs a separate estimator, scorer, refiner, CUDA rasterizer,
input queue, last pose, initialization dataset, and output directory. Frames
are submitted through size-one latest-frame queues. The estimates are never
fused or selected between. `--foundationpose-root` selects the complete
external FoundationPose installation used for imports, weights, compiled
components, and `box.obj`; the repository-local `foundationpose/` tree is only
a lightweight development/reference copy.

Frame convention is `A_T_B`: map B coordinates into A. The exact chains are:

```text
B_T_C(q)       = pelvis_T_d435(q) @ D_T_C_ROS
K_T_B_vision   = inv(C_T_K) @ inv(B_T_C(q))
K_T_V          = K_T_B_vision @ inv(T_T_B) @ inv(V_T_T)  [30-frame mean]
K_T_B          = K_T_V @ V_T_T @ T_T_B
K_T_C          = K_T_B @ B_T_C(q)
K_T_box_ext    = K_T_E @ E_T_box
K_T_box_g1     = K_T_C @ C_T_box
```

The UI shows the external camera, pelvis, onboard camera, VIVE tracker, and
both labelled boxes in K. It reports Euclidean translation separation in mm,
raw SO(3) geodesic rotation separation in degrees, and the minimum rotation
separation over the eight proper D4 symmetries of the 0.40 x 0.30 x 0.30 m
cuboid. Available poses are saved under
`integration/outputs/latest/transforms/`.

Hardware-free mock/static test:

```bash
cd /home/chris/Chris/g1_vision_system
python3 -m unittest -v integration.test_transforms
python3 -m py_compile integration/*.py
```

Intended real runtime (from an environment containing both the existing VIVE
stack and FoundationPose dependencies/CUDA models):

```bash
cd /home/samsung/Chris/g1_box_tracking
G1_IFACE=enx6c1ff7bf07c7 python integration/g1_unified_vision.py \
  --foundationpose-root /home/samsung/Chris/FoundationPose \
  --tracker 0d:e1:7b:f0 \
  --tracker-tf vive/g1_tracker_system/calibration/T_tracker_from_g1_root.txt \
  --g1-init-dir /home/samsung/Chris/FoundationPose/g1/data/live_init \
  --external-init-dir /path/to/external_live_init
```

The FoundationPose runtime root and both initialization paths are required
runtime arguments. No weights, compiled artifacts, or RGB/depth/mask datasets
are stored or assumed to exist in this source repository. The runtime loads
FoundationPose from `/home/samsung/Chris/FoundationPose` and resolves the mesh
as `/home/samsung/Chris/FoundationPose/box.obj`. The G1 initialization path
above already exists on the `.124` machine. A separate external-camera
initialization dataset still needs to be created, and
`/path/to/external_live_init` must be replaced with its real location.

Each init directory must contain its own `cam_K.txt` and
`rgb/000000.png`, `depth/000000.png`, and `masks/000000.png`. Depth init PNGs
use the existing FoundationPose convention of uint16 millimetres. Press `I` to
manually restart common-world alignment, `X` to clear it, and `Q` to quit.

## Runtime notes

FoundationPose model weights and large datasets are not stored here.

Working FoundationPose runtime on company PC:
`/home/samsung/miniconda3/envs/foundationpose5080`

Original full FoundationPose runtime/data:
`/home/samsung/Chris/FoundationPose`

Current G1 interface:
`enx6c1ff7bf07c7`

External D435i serial:
`262322070500`

G1 camera endpoint:
`tcp://192.168.123.164:5555`
