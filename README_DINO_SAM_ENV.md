# FoundationPose + DINO + SAM2 Setup

## Overview

This project currently uses a split setup:

- `FoundationPose` + `SAM2` run in the `foundationpose` conda environment.
- Grounding DINO runs through an external Python environment because the `FoundationPose` stack is pinned to `torch 2.0.0`, while recent `transformers` releases expect newer Torch.

The current code already handles this split automatically through
[`phantom/phantom/detectors/detector_dino.py`](/projects/zaijia001/phantom/phantom/detectors/detector_dino.py).

## Recommended Runtime Layout

### Main environment

- Conda env: `/projects/zaijia001/conda_envs/foundationpose`
- Purpose:
  - `FoundationPose`
  - `PyTorch3D`
  - `nvdiffrast`
  - `SAM2`
  - main inference scripts

### External DINO Python

Grounding DINO is tried in this order:

1. `PHANTOM_DINO_PYTHON`
2. `/projects/_ssd/zaijia001/RoboTwin/policy/pi0/.venv/bin/python`
3. `/projects/zaijia001/conda_envs/openpi/bin/python`

If one of those can run:

```bash
python -c "from transformers import pipeline; print('ok')"
```

then DINO should work.

## Runtime Activation

Use:

```bash
source /projects/zaijia001/FoundationPose/source_foundationpose_env.sh
```

This script loads:

- `CUDA/11.8.0`
- `GCCcore/13.2.0`
- `libglvnd/1.7.0`

and activates:

- `/projects/zaijia001/conda_envs/foundationpose`

## Fresh Install Steps

### 1. Create the main env

```bash
conda create -n foundationpose python=3.9 -y
conda activate foundationpose
```

### 2. Install core FoundationPose deps

Inside `/projects/zaijia001/FoundationPose`:

```bash
python -m pip install -r requirements.txt
```

If needed:

```bash
conda install conda-forge::eigen=3.4.0
```

### 3. Install `nvdiffrast`

Use a compiler/CUDA combination compatible with the FoundationPose Torch stack:

```bash
module purge
module load GCC/12.3.0
module load CUDA/11.8.0

python -m pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
```

### 4. Install `PyTorch3D`

```bash
python -m pip install --quiet --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
```

### 5. Install `SAM2` into the same env

Do not let `pip` upgrade Torch.

```bash
cd /projects/zaijia001/phantom/submodules/sam2
SAM2_BUILD_CUDA=0 python -m pip install -e . --no-deps --no-build-isolation
```

Notes:

- This repo was patched locally so `sam2` accepts Python 3.9.
- The local package metadata was also relaxed to avoid forcing a Torch upgrade.

### 6. Install light HuggingFace deps in the main env

These are needed for helper code paths, even though DINO itself is recommended to run in an external env:

```bash
python -m pip install hydra-core
```

`transformers` inside `foundationpose` is not the recommended path for DINO inference here.

## DINO Environment Options

### Option A: Reuse existing envs

Preferred if already available:

- `/projects/_ssd/zaijia001/RoboTwin/policy/pi0/.venv`
- `/projects/zaijia001/conda_envs/openpi`

Verify one of them:

```bash
/projects/_ssd/zaijia001/RoboTwin/policy/pi0/.venv/bin/python -c "from transformers import pipeline; print('ok')"
```

### Option B: Create a dedicated DINO env

```bash
conda create -n grounding_dino python=3.10 -y
conda activate grounding_dino
python -m pip install torch torchvision transformers accelerate pillow
python -c "from transformers import pipeline; print('ok')"
```

Then export:

```bash
export PHANTOM_DINO_PYTHON=/path/to/grounding_dino/bin/python
```

## Checkpoints and Weights

Required files:

- FoundationPose weights:
  - `/projects/zaijia001/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth`
  - `/projects/zaijia001/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth`
- SAM2 checkpoint:
  - `/projects/zaijia001/phantom/submodules/sam2/checkpoints/sam2_hiera_large.pt`

The current machine already has symlinks prepared from:

- `/projects/_ssd/zaijia001/OnePoseviaGen/checkpoints/OnePoseViaGen/FoundationPose/`
- `/projects/_ssd/zaijia001/OnePoseviaGen/checkpoints/OnePoseViaGen/SAM2/`

## Verification

### Main env

```bash
source /projects/zaijia001/FoundationPose/source_foundationpose_env.sh
python -c "import torch, cv2, pytorch3d, nvdiffrast; print(torch.cuda.is_available())"
```

### DINO fallback env

```bash
/projects/_ssd/zaijia001/RoboTwin/policy/pi0/.venv/bin/python -c "from transformers import pipeline; print('dino ok')"
```

### Main script

```bash
cd /projects/zaijia001/FoundationPose
python run_realr1_dino_sam.py --help
python run_realr1_dino_sam_batch.py --help
```

## Single Video Run

```bash
python /projects/zaijia001/FoundationPose/run_realr1_dino_sam.py \
  --data_dir /projects/zaijia001/R1/hand/d_pour_low \
  --video_id 1 \
  --mesh_file /projects/_hdd/zaijia/R1/obj_mesh/bottle.obj \
  --output_dir /projects/zaijia001/R1/object_pose/d_pour_low_bottle_1 \
  --prompt bottle \
  --save_video 1 \
  --save_mesh_overlay_video 1 \
  --save_bbox_overlay_video 1
```

## Batch Run

```bash
python /projects/zaijia001/FoundationPose/run_realr1_dino_sam_batch.py \
  --data_dir /projects/zaijia001/R1/hand/d_pour_low \
  --mesh_file /projects/_hdd/zaijia/R1/obj_mesh/bottle.obj \
  --output_root /projects/zaijia001/R1/object_pose \
  --prompt bottle \
  --save_video 1 \
  --save_mesh_overlay_video 1 \
  --save_bbox_overlay_video 1
```

Default output naming:

- `{data_dir.name}_{prompt}_{video_id}`

Example:

- `d_pour_low_bottle_5`

## Notes

- `run_realr1_dino_sam.py` prefers `depth_{id}/` raw depth frames.
- `depth_vis_{id}.mp4` is only for preview.
- The scanned bottle mesh currently has missing texture image files; the code was patched to fall back to a pure-color mesh instead of crashing.
