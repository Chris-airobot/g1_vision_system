# FoundationPose environment setup (Conda)

This file records the environment used for the RGB-D box-tracking implementation.

## Tested stack

- Ubuntu Linux
- Python 3.11
- CUDA Toolkit 12.8
- PyTorch 2.7.1 + cu128
- torchvision 0.22.1 + cu128
- torchaudio 2.7.1 + cu128

## 1. Clone the repository

```bash
git clone https://github.com/Chris-airobot/FoundationPose.git
cd FoundationPose
```

## 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate foundationpose
```

## 3. Install PyTorch CUDA 12.8

```bash
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

## 4. Configure CUDA paths

Example for a standard CUDA 12.8 installation:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export C_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
```

Change `CUDA_HOME` if your CUDA toolkit is installed elsewhere.

## 5. Build/install GPU dependencies

```bash
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git"

python -m pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast.git"

python -m pip install -r requirements.txt
bash build_all_conda.sh
```

## 6. Verify the environment

```bash
python check_env.py
```

## FoundationPose weights

The expected layout is:

```text
weights/
  2023-10-28-18-33-37/   # refiner
    model_best.pth
    config.yml
  2024-01-11-20-02-45/   # scorer
    model_best.pth
    config.yml
```

Official pretrained weights:

https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing

If extraction creates an extra nested directory, move the timestamp directory so it sits directly under `weights/`.

## Capture a RealSense box sequence

Install the RealSense Python wrapper if needed:

```bash
conda activate foundationpose
python -m pip install pyrealsense2
```

List connected devices:

```bash
python capture_realsense_box.py --list-devices
```

Record a sequence:

```bash
python capture_realsense_box.py \
  --serial REPLACE_WITH_SERIAL \
  --output-dir demo_data/box_test
```

The recorder aligns depth to color, stores depth in 16-bit millimetres, saves camera intrinsics to `cam_K.txt`, and creates the first-frame object mask interactively.

Run the recorded sequence:

```bash
python run_demo.py \
  --mesh_file box.obj \
  --test_scene_dir demo_data/box_test \
  --debug 2 \
  --debug_dir debug_box_test
```

## Run the standard upstream demo

After downloading the official demo data and weights:

```bash
conda activate foundationpose
python run_demo.py
```
