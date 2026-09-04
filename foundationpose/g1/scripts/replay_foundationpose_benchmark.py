import os, sys, time
from collections import deque
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from estimater import *

DATA = ROOT / 'g1/data/fps_benchmark'
OUT = ROOT / 'g1/results/fps_benchmark_replay'
K = np.loadtxt(DATA / 'cam_K.txt').reshape(3, 3)
TRACK_ITERS = 1
VIS_EVERY = 3


def depth_m(d):
    d = d.astype(np.float32) * 0.001
    d[(d < 0.001) | (d > 10.0)] = 0
    return d


def fps_from_times(ts):
    if len(ts) < 2:
        return 0.0
    dt = ts[-1] - ts[0]
    return (len(ts) - 1) / dt if dt > 0 else 0.0


def load_rgb(path):
    # cv2.imread returns BGR, while FoundationPose expects RGB.
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f'Failed to read RGB frame: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_ui(rgb):
    # OpenCV display expects BGR; inference keeps RGB.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    shown = bgr.copy(); pts = []
    win = 'Offline benchmark: draw first-frame mask'

    def mouse(e, x, y, flags, param):
        nonlocal shown
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y)); shown = bgr.copy()
            if len(pts) > 1:
                cv2.polylines(shown, [np.array(pts)], False, (0, 255, 0), 2)
            for q in pts:
                cv2.circle(shown, q, 4, (0, 0, 255), -1)

    cv2.namedWindow(win); cv2.setMouseCallback(win, mouse)
    print('Click around the box; ENTER accept; R reset; ESC cancel')
    while True:
        cv2.imshow(win, shown)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(pts) >= 3:
            m = np.zeros(rgb.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.array(pts)], 1)
            cv2.destroyWindow(win)
            return m.astype(bool)
        if key == ord('r'):
            pts.clear(); shown = bgr.copy()
        if key == 27:
            cv2.destroyWindow(win); return None


def main():
    set_logging_format(); set_seed(0); os.makedirs(OUT, exist_ok=True)
    rgb_files = sorted((DATA / 'rgb').glob('*.png'))
    dep_files = sorted((DATA / 'depth').glob('*.png'))
    n = min(len(rgb_files), len(dep_files))
    if n < 2:
        raise RuntimeError(f'Need recorded RGB-D frames; found rgb={len(rgb_files)}, depth={len(dep_files)}')
    print(f'OFFLINE REPLAY: {n} RGB-D frames, track iterations={TRACK_ITERS}, vis every {VIS_EVERY}')

    mesh = trimesh.load(ROOT / 'box.obj')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    center_tf = np.linalg.inv(to_origin)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug_dir=str(OUT), debug=0,
        glctx=dr.RasterizeCudaContext())

    rgb0 = load_rgb(rgb_files[0])
    dep0 = depth_m(cv2.imread(str(dep_files[0]), cv2.IMREAD_UNCHANGED))
    mask_dir = DATA / 'masks'; mask_dir.mkdir(exist_ok=True)
    mask_file = mask_dir / '000000.png'
    if mask_file.exists():
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED).astype(bool)
        print('Using saved first-frame mask:', mask_file)
    else:
        mask = mask_ui(rgb0)
        if mask is None:
            print('Cancelled.'); return
        cv2.imwrite(str(mask_file), (mask.astype(np.uint8) * 255))
        print('Saved first-frame mask:', mask_file)

    print('Registering first recorded frame...')
    pose = est.register(K=K, rgb=rgb0, depth=dep0, ob_mask=mask, iteration=5)
    print('Registration complete. Starting offline tracking.')

    pose_times = deque(maxlen=120)
    track_times = deque(maxlen=120)
    start = time.perf_counter()

    for i in range(1, n):
        rgb = load_rgb(rgb_files[i])
        dep = depth_m(cv2.imread(str(dep_files[i]), cv2.IMREAD_UNCHANGED))
        t0 = time.perf_counter()
        pose = est.track_one(rgb=rgb, depth=dep, K=K, iteration=TRACK_ITERS)
        track_ms = (time.perf_counter() - t0) * 1000.0
        now = time.perf_counter(); pose_times.append(now); track_times.append(track_ms)
        pose_fps = fps_from_times(pose_times)

        if i % VIS_EVERY == 0:
            cp = pose @ center_tf
            vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=cp, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=cp, scale=0.1, K=K, thickness=3,
                                transparency=0, is_input_rgb=True)
            show = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            txt = f'offline pose {pose_fps:.1f} FPS | track {track_ms:.1f} ms | frame {i}/{n-1}'
            cv2.putText(show, txt, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.imshow('FoundationPose OFFLINE benchmark', show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if i % 100 == 0:
            print(f'frame={i} | pose FPS={pose_fps:.2f} | track={track_ms:.1f} ms')

    elapsed = time.perf_counter() - start
    mean_track = float(np.mean(track_times)) if track_times else 0.0
    processed = i
    print(f'DONE | processed={processed} | wall FPS={processed/elapsed:.2f} | recent pose FPS={fps_from_times(pose_times):.2f} | mean recent track={mean_track:.1f} ms')
    np.savetxt(OUT / 'latest_pose.txt', pose)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
