import os, sys, time
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from estimater import *

DATA = ROOT / 'g1/data/fps_benchmark'
K = np.loadtxt(DATA / 'cam_K.txt').reshape(3, 3)
TRACK_ITERS = 1
WARMUP = 20
BENCH = 300


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f'Failed to read {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def depth_m(path):
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
    d[(d < 0.001) | (d > 10.0)] = 0
    return d


def main():
    set_logging_format(); set_seed(0)
    rgb_files = sorted((DATA/'rgb').glob('*.png'))
    dep_files = sorted((DATA/'depth').glob('*.png'))
    n = min(len(rgb_files), len(dep_files), WARMUP + BENCH + 1)
    if n < WARMUP + 30:
        raise RuntimeError(f'Not enough frames: {n}')

    print(f'Preloading {n} RGB-D frames into RAM...')
    rgbs = [load_rgb(p) for p in rgb_files[:n]]
    deps = [depth_m(p) for p in dep_files[:n]]
    print('Preload complete.')

    mesh = trimesh.load(ROOT/'box.obj')
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug_dir='/tmp/fp_core_bench', debug=0,
        glctx=dr.RasterizeCudaContext())

    mask_file = DATA/'masks/000000.png'
    if not mask_file.exists():
        raise RuntimeError(f'Missing {mask_file}; run offline replay once first.')
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED).astype(bool)

    print('Registering...')
    est.register(K=K, rgb=rgbs[0], depth=deps[0], ob_mask=mask, iteration=5)

    print(f'Warmup: {WARMUP} frames')
    for i in range(1, WARMUP + 1):
        est.track_one(rgb=rgbs[i], depth=deps[i], K=K, iteration=TRACK_ITERS)

    times = []
    torch.cuda.synchronize()
    wall0 = time.perf_counter()
    end = min(n, WARMUP + 1 + BENCH)
    for i in range(WARMUP + 1, end):
        t0 = time.perf_counter()
        est.track_one(rgb=rgbs[i], depth=deps[i], K=K, iteration=TRACK_ITERS)
        times.append((time.perf_counter() - t0) * 1000.0)
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall0

    a = np.asarray(times)
    fps = len(a) / wall
    print(f'CORE BENCH | frames={len(a)} | FPS={fps:.2f} | mean={a.mean():.2f} ms | p50={np.percentile(a,50):.2f} ms | p95={np.percentile(a,95):.2f} ms')
    print('60 FPS requires <=16.67 ms/frame end-to-end.')

if __name__ == '__main__':
    main()
