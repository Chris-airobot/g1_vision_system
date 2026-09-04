# G1 Box Tracking

Unified development repo for Samsung Unitree G1 box perception/tracking.

## Components

- `vive/`
  - VIVE Ultimate tracker interface
  - G1 root/common-world tracking
  - ChArUco calibration
  - G1 camera/root visualization

- `foundationpose/`
  - Lightweight FoundationPose reference source and G1 box scripts
  - heavy weights/data are intentionally excluded

- `integration/`
  - unified G1 + VIVE + dual-camera + FoundationPose system

## Common frames

- `E`: fixed world frame and external D435i optical frame
- `V`: VIVE tracking world
- `B`: G1 pelvis/root
- `C`: G1 onboard camera
- `T`: VIVE tracker
- `O`: box

External camera box:
`E_T_O_ext = E_T_O`

G1 camera box:
`E_T_O_g1 = E_T_V @ V_T_T @ T_T_B @ B_T_C(q) @ C_T_O`

where:
`E_T_C = E_T_V @ V_T_T @ T_T_B @ B_T_C(q)`

The two camera trackers remain independent and their valid estimates are fused in E.

## Unified integration runtime

`integration/g1_unified_vision.py` loads the saved `E_T_V`, existing VIVE
body-axis remap, calibrated `T_T_B`, and rev1.0 URDF waist FK. ChArUco is not
used or required at runtime. Camera acquisition and FoundationPose inference
run outside the visualization loop.

The external and onboard streams each have their own `FoundationPoseWorker`.
Each worker constructs a separate estimator, scorer, refiner, CUDA rasterizer,
input queue, last pose, initialization dataset, and output directory. Each
stream has a latest-frame queue and its own submission throttle. Invalid
streams retry at a lower rate. A worker can be periodically reseeded from the
other camera's valid world pose, with conversion to FoundationPose's internal
centered-mesh `pose_last`. `--foundationpose-root` selects the complete
external FoundationPose installation used for imports, weights, compiled
components, and `box.obj`; the repository-local `foundationpose/` tree is only
a lightweight development/reference copy.

Frame convention is `A_T_B`: map B coordinates into A. The runtime chains are:

```text
B_T_C(q)       = pelvis_T_d435(q) @ D_T_C_ROS
E_T_C          = E_T_V @ V_T_T @ T_T_B @ B_T_C(q)
E_T_box_ext    = E_T_box
E_T_box_g1     = E_T_C @ C_T_box
```

Pose visibility is classified as `TRACKING`, `PARTIAL`, or `LOST`. For the
exact RGB-D frame associated with each FoundationPose result, the validator
ray-casts the nearest expected 30 cm cube surface and compares observed depth
pixel-by-pixel using a 25 mm or 1.5% depth tolerance, whichever is larger.
Image overlap, depth coverage, agreeing surface pixels, closer occluding
geometry, and farther/background geometry contribute to the classification
and quality. `TRACKING` and `PARTIAL` participate in fusion; `LOST` never does.

When both estimates are usable, translation is quality-weighted and rotation
is averaged after resolving the closest of 24 proper cube symmetries. A lone
usable estimate passes through. If both cameras are `LOST`, the current fused
pose immediately becomes invalid: source is `NONE`, the fused wireframe is not
drawn, and saved current-pose files are removed. Available current poses are
saved under `integration/outputs/latest/transforms/`.

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
  --external-vive-tf vive/g1_tracker_system/calibration/T_external_from_vive_world.txt \
  --g1-init-dir /home/samsung/Chris/FoundationPose/g1/data/live_init \
  --external-init-dir /path/to/external_live_init
```

The FoundationPose runtime root and both initialization paths are required
runtime arguments. No weights, compiled artifacts, or RGB/depth/mask datasets
are stored or assumed to exist in this source repository. The runtime loads
FoundationPose from `/home/samsung/Chris/FoundationPose` and resolves the mesh
as `/home/samsung/Chris/FoundationPose/box.obj`. Both synchronized camera
initialization datasets remain outside this Git repository.

Each init directory must contain its own `cam_K.txt` and
`rgb/000000.png`, `depth/000000.png`, and `masks/000000.png`. Depth init PNGs
use the existing FoundationPose convention of uint16 millimetres. Press `L` to
reload the tracker mount transform, `R` to clear the trail, and `Q` to quit.

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
