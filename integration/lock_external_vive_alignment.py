#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIVE_SCRIPTS = ROOT / "vive" / "g1_tracker_system" / "scripts"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VIVE_SCRIPTS))

import g1_common_frame_visualizer_interactive as base
import g1_hybrid_tracker_visualizer as hybrid


CALIB = (
    ROOT
    / "vive/g1_tracker_system/calibration"
    / "dual_camera_charuco_results"
    / "T_external_from_g1_camera.txt"
)

TRACKER_TF = (
    ROOT
    / "vive/g1_tracker_system/calibration"
    / "T_tracker_from_g1_root.txt"
)

OUT = (
    ROOT
    / "vive/g1_tracker_system/calibration"
    / "T_external_from_vive_world.txt"
)

N = 30


def main():
    print("============================================================")
    print("LOCK EXTERNAL CAMERA <-> VIVE WORLD")
    print("============================================================")
    print("KEEP G1 AND EXTERNAL CAMERA COMPLETELY STILL.")
    print()

    if not CALIB.exists():
        raise RuntimeError(f"Missing dual-camera calibration: {CALIB}")

    E_T_C = np.loadtxt(CALIB)
    T_T_B = hybrid.load_T_T_B(TRACKER_TF)

    if T_T_B is None:
        raise RuntimeError(f"Missing tracker mount transform: {TRACKER_TF}")

    print("Dual-camera calibration:", CALIB)
    print("Tracker mount:", TRACKER_TF)
    print()

    low = base.LowStateReader()
    vive = hybrid.ViveReader("0d:e1:7b:f0")

    samples = []
    last_vive_time = -1.0

    try:
        while len(samples) < N:
            now = time.monotonic()

            q, mode_machine, low_time = low.get()
            V_T_T, status, hz, dev, vive_time, vive_error = vive.get()

            low_ok = (
                q is not None
                and now - low_time < hybrid.LOWSTATE_STALE_SEC
                and mode_machine == 5
            )

            vive_ok = (
                V_T_T is not None
                and status == "OK"
                and dev == "0d:e1:7b:f0"
                and now - vive_time < hybrid.VIVE_STALE_SEC
            )

            if not low_ok or not vive_ok:
                print(
                    f"\rwaiting... low={low_ok} "
                    f"vive={vive_ok} status={status} hz={hz:.1f}",
                    end="",
                    flush=True,
                )
                time.sleep(0.01)
                continue

            # Only use a new VIVE sample.
            if vive_time == last_vive_time:
                time.sleep(0.002)
                continue

            last_vive_time = vive_time

            # pelvis <- G1 optical camera
            B_T_C = base.pelvis_T_d435(q) @ base.D_T_C_ROS

            # external <- pelvis at the calibration pose
            E_T_B = E_T_C @ base.invT(B_T_C)

            # Current runtime convention:
            #
            # E_T_B = E_T_V @ V_T_T @ T_T_B
            #
            # therefore:
            #
            # E_T_V = E_T_B @ inv(T_T_B) @ inv(V_T_T)
            E_T_V = (
                E_T_B
                @ base.invT(T_T_B)
                @ base.invT(V_T_T)
            )

            samples.append(E_T_V)

            print(
                f"\rcollecting {len(samples):02d}/{N} "
                f"| VIVE {hz:.1f} Hz",
                end="",
                flush=True,
            )

            time.sleep(0.01)

    finally:
        vive.stop()

    E_T_V = hybrid.average_T(samples)

    errors = np.asarray([
        hybrid.pose_error(E_T_V, T)
        for T in samples
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT, E_T_V, fmt="%.10f")

    print()
    print()
    print("============================================================")
    print("VIVE ALIGNMENT LOCKED")
    print("============================================================")
    print()
    print("T_external_from_vive_world =")
    print(E_T_V)
    print()
    print(
        f"30-frame spread: "
        f"{errors[:,0].mean():.2f} +/- {errors[:,0].std():.2f} mm"
    )
    print(
        f"rotation spread: "
        f"{errors[:,1].mean():.3f} +/- {errors[:,1].std():.3f} deg"
    )
    print()
    print("Saved:")
    print(OUT)
    print()
    print("BOARD IS NO LONGER REQUIRED.")
    print("============================================================")


if __name__ == "__main__":
    main()
