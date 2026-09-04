import cv2
import numpy as np
import os

img_path = os.path.expanduser("~/g1_recording/rgb/000000.png")
out_dir = os.path.expanduser("~/g1_recording/masks")
out_path = os.path.join(out_dir, "000000.png")

os.makedirs(out_dir, exist_ok=True)

img = cv2.imread(img_path)
display = img.copy()
points = []

def mouse(event, x, y, flags, param):
    global display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        display = img.copy()

        if len(points) > 1:
            cv2.polylines(
                display,
                [np.array(points)],
                False,
                (0, 255, 0),
                2
            )

        for p in points:
            cv2.circle(display, p, 4, (0, 0, 255), -1)

cv2.namedWindow("Draw box mask")
cv2.setMouseCallback("Draw box mask", mouse)

print("Left click around the object.")
print("Press ENTER to save.")
print("Press R to reset.")
print("Press ESC to cancel.")

while True:
    cv2.imshow("Draw box mask", display)
    key = cv2.waitKey(20) & 0xFF

    if key == 13:  # ENTER
        if len(points) < 3:
            print("Need at least 3 points.")
            continue

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points)], 255)
        cv2.imwrite(out_path, mask)

        print("Saved:", out_path)
        break

    elif key == ord("r"):
        points = []
        display = img.copy()

    elif key == 27:
        print("Cancelled.")
        break

cv2.destroyAllWindows()
