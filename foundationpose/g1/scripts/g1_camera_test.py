import zmq
import msgpack
import cv2
import numpy as np
import base64

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.setsockopt_string(zmq.SUBSCRIBE, "")
sock.connect("tcp://192.168.123.164:5555")

print("Waiting for camera frame...")

data = msgpack.unpackb(sock.recv(), raw=False)

print("Camera keys:", list(data["images"].keys()))

for name, value in data["images"].items():
    if isinstance(value, str):
        buf = base64.b64decode(value)
    else:
        buf = value

    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    
    if name == "ego_view":
    	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    print(name, img.shape, img.dtype)
    cv2.imwrite(f"{name}.png", img)

print("Saved camera images.")
