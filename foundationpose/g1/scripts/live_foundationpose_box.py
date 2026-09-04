import os, sys, base64, time, threading
from collections import deque
from pathlib import Path
import cv2, zmq, msgpack, numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from estimater import *

ENDPOINT='tcp://192.168.123.164:5555'
INIT=ROOT/'g1/data/live_init'
OUT=ROOT/'g1/results/live_foundationpose'
K=np.loadtxt(INIT/'cam_K.txt').reshape(3,3)

# Tracking runs on every received frame. Expensive X11 visualization runs less often.
VIS_EVERY=3
POSE_SAVE_INTERVAL=1.0
TRACK_ITERS=1

cam_times=deque(maxlen=120)
cam_lock=threading.Lock()


def decode(payload):
    data=msgpack.unpackb(payload,raw=False)
    out={}
    for k,v in data['images'].items():
        b=base64.b64decode(v) if isinstance(v,str) else v
        out[k]=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED)
    return out


def depth_m(d):
    d=d.astype(np.float32)
    d*=0.001
    d[(d<0.001)|(d>10.0)]=0
    return d


def fps_from_times(times):
    if len(times)<2: return 0.0
    dt=times[-1]-times[0]
    return (len(times)-1)/dt if dt>0 else 0.0


def camera_fps_monitor():
    ctx=zmq.Context(); sock=ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE,'')
    sock.setsockopt(zmq.CONFLATE,1)
    sock.connect(ENDPOINT)
    while True:
        try:
            sock.recv()
            with cam_lock:
                cam_times.append(time.perf_counter())
        except Exception:
            break


def mask_ui(rgb):
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR); shown=bgr.copy(); pts=[]
    win='Reinitialize mask'
    def mouse(e,x,y,f,p):
        nonlocal shown
        if e==cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y)); shown=bgr.copy()
            if len(pts)>1: cv2.polylines(shown,[np.array(pts)],False,(0,255,0),2)
            for q in pts: cv2.circle(shown,q,4,(0,0,255),-1)
    cv2.namedWindow(win); cv2.setMouseCallback(win,mouse)
    print('Left click box; ENTER accept; R reset; ESC cancel')
    while True:
        cv2.imshow(win,shown); key=cv2.waitKey(20)&0xFF
        if key==13 and len(pts)>=3:
            m=np.zeros(rgb.shape[:2],np.uint8); cv2.fillPoly(m,[np.array(pts)],1)
            cv2.destroyWindow(win); return m.astype(bool)
        if key==ord('r'): pts.clear(); shown=bgr.copy()
        if key==27: cv2.destroyWindow(win); return None


def main():
    set_logging_format(); set_seed(0); os.makedirs(OUT,exist_ok=True)
    mesh=trimesh.load(ROOT/'box.obj')
    to_origin,extents=trimesh.bounds.oriented_bounds(mesh)
    center_tf=np.linalg.inv(to_origin)
    bbox=np.stack([-extents/2,extents/2],axis=0).reshape(2,3)
    est=FoundationPose(model_pts=mesh.vertices,model_normals=mesh.vertex_normals,mesh=mesh,
        scorer=ScorePredictor(),refiner=PoseRefinePredictor(),debug_dir=str(OUT),debug=0,
        glctx=dr.RasterizeCudaContext())

    bgr=cv2.imread(str(INIT/'rgb/000000.png'))
    rgb0=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    dep0=depth_m(cv2.imread(str(INIT/'depth/000000.png'),-1))
    mask=cv2.imread(str(INIT/'masks/000000.png'),-1).astype(bool)
    pose=est.register(K=K,rgb=rgb0,depth=dep0,ob_mask=mask,iteration=5)
    np.savetxt(OUT/'init_pose.txt',pose)

    threading.Thread(target=camera_fps_monitor,daemon=True).start()

    ctx=zmq.Context(); sock=ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE,''); sock.setsockopt(zmq.CONFLATE,1)
    sock.connect(ENDPOINT)
    print('FAST LIVE: q quit, r redraw mask/re-register')
    print(f'track iterations={TRACK_ITERS}; visualization every {VIS_EVERY} pose frames')

    i=0
    pose_times=deque(maxlen=90)
    last_report=time.perf_counter()
    last_pose_save=last_report

    try:
        while True:
            im=decode(sock.recv())
            if 'ego_view' not in im or 'ego_view_depth' not in im: continue
            rgb=im['ego_view']; dep=depth_m(im['ego_view_depth'])

            t0=time.perf_counter()
            pose=est.track_one(rgb=rgb,depth=dep,K=K,iteration=TRACK_ITERS)
            track_ms=(time.perf_counter()-t0)*1000.0

            now=time.perf_counter(); pose_times.append(now)
            pose_fps=fps_from_times(pose_times)
            with cam_lock:
                camera_fps=fps_from_times(list(cam_times))

            # Disk I/O used to happen every frame. Save only once per second.
            if now-last_pose_save>=POSE_SAVE_INTERVAL:
                np.savetxt(OUT/'latest_pose.txt',pose)
                last_pose_save=now

            if now-last_report>=1.0:
                print(f'camera-arrival FPS={camera_fps:.2f} | pose-output FPS={pose_fps:.2f} | '
                      f'track={track_ms:.1f} ms | iterations={TRACK_ITERS} | vis_every={VIS_EVERY}')
                last_report=now

            # X11 rendering/display is intentionally decimated; pose inference is not.
            if i%VIS_EVERY==0:
                cp=pose@center_tf
                vis=draw_posed_3d_box(K,img=rgb,ob_in_cam=cp,bbox=bbox)
                vis=draw_xyz_axis(vis,ob_in_cam=cp,scale=0.1,K=K,thickness=3,
                                  transparency=0,is_input_rgb=True)
                show=vis[...,::-1].copy()
                text=f'cam {camera_fps:.1f} | pose {pose_fps:.1f} FPS | track {track_ms:.0f} ms | vis 1/{VIS_EVERY}'
                cv2.putText(show,text,(15,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
                cv2.imshow('G1 FoundationPose Live FAST',show)

            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break
            if key==ord('r'):
                m=mask_ui(rgb)
                if m is not None:
                    pose=est.register(K=K,rgb=rgb,depth=dep,ob_mask=m,iteration=5)
                    pose_times.clear()
            i+=1
    finally:
        np.savetxt(OUT/'latest_pose.txt',pose)
        cv2.destroyAllWindows(); sock.close(0); ctx.term()

if __name__=='__main__': main()
