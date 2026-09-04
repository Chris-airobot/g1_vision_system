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
