import os
import cv2
import zmq
import msgpack
import base64
import numpy as np
import time

OUT = os.path.expanduser("~/g1_recording")
RGB_DIR = os.path.join(OUT, "rgb")
DEPTH_DIR = os.path.join(OUT, "depth")

os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.setsockopt_string(zmq.SUBSCRIBE, "")
sock.connect("tcp://192.168.123.164:5555")

print("Waiting for camera...")
print("Press q in the camera window to stop recording.")

i = 0
t0 = time.time()

while True:
    data = msgpack.unpackb(sock.recv(), raw=False)

    images = {}
    for name, value in data["images"].items():
        buf = base64.b64decode(value) if isinstance(value, str) else value
        images[name] = cv2.imdecode(
            np.frombuffer(buf, np.uint8),
            cv2.IMREAD_UNCHANGED
        )

    rgb = images["ego_view"]
    depth = images["ego_view_depth"]

    # Fix RGB/BGR issue
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Save FoundationPose-compatible frames
    cv2.imwrite(os.path.join(RGB_DIR, f"{i:06d}.png"), rgb)
    cv2.imwrite(os.path.join(DEPTH_DIR, f"{i:06d}.png"), depth)

    # Depth only for visualization
    depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    preview = np.hstack((rgb, depth_vis))

    cv2.putText(
        preview,
        f"Frame {i}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("G1 RGB + Depth", preview)

    i += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

elapsed = time.time() - t0

cv2.destroyAllWindows()
sock.close()
ctx.term()

print(f"Recorded {i} frames")
print(f"Duration: {elapsed:.1f} s")
print(f"Average FPS: {i / elapsed:.2f}")
print(f"Saved to: {OUT}")
