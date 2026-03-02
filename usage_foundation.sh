# 推荐运行顺序：
# 最省事的方式：
#   source /projects/zaijia001/FoundationPose/source_foundationpose_env.sh
#
# 1) 如果 prompt 里还有 `(openpi)`，先执行 `conda deactivate` 或退出对应 venv，避免抢占 python/pip
# 2) 再激活 foundationpose:
#    conda activate /projects/zaijia001/conda_envs/foundationpose
# 3) 运行前加载模块（先清掉旧模块，避免 GCC/GCCcore 冲突）：
#    module purge
#    module load CUDA/11.8.0
#    module load GCCcore/13.2.0
#    module load libglvnd/1.7.0
# 4) 运行需要启动：
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH"
#
# 检查解释器是否正确：
#   which python
#   python -c "import sys; print(sys.executable)"
# 应该指向 /projects/zaijia001/conda_envs/foundationpose/bin/python


# create conda environment
conda create -n foundationpose python=3.9

# activate conda environment
conda activate foundationpose

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# install dependencies
python -m pip install -r requirements.txt

# 如果缺 DINO 依赖，固定到和 torch 2.0.0 兼容的版本：
python -m pip install "transformers==4.41.2" "accelerate==0.30.1" hydra-core

# Install NVDiffRast
# python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
module purge
module load GCC/12.3.0
# module load CUDA/12.1.1
# module load cuDNN/8.9.2.26-CUDA-12.1.1
module load CUDA/11.8.0
module avail GCC

python -m pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
module unload GCC/12.3.0
module load GCC/11.4.0

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# SAM2:
# 不要直接 `pip install -e .`，否则大概率升级 torch，破坏 FoundationPose + PyTorch3D 组合。
# 本地已经把 sam2 的 python_requires 从 >=3.10 调整到 >=3.9，以兼容 foundationpose 环境。
# 在 foundationpose 环境里建议：
#   cd /projects/zaijia001/phantom/submodules/sam2
#   SAM2_BUILD_CUDA=0 python -m pip install -e . --no-deps --no-build-isolation
# checkpoint 需要额外放到：
#   /projects/zaijia001/phantom/submodules/sam2/checkpoints/sam2_hiera_large.pt

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

# FoundationPose 权重需要放到：
# /projects/zaijia001/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth
# /projects/zaijia001/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth
# 本机现成权重来源：
# /projects/_ssd/zaijia001/OnePoseviaGen/checkpoints/OnePoseViaGen/FoundationPose/


# 启动
conda deactivate   # 先退掉 openpi/其它环境，直到 prompt 里没有它们

module load GCC/12.3.0
module load CUDA/11.8.0
module load GCCcore/13.2.0
module load libglvnd/1.7.0

conda activate /projects/zaijia001/conda_envs/foundationpose
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH"

which python
python -c "import sys; print(sys.executable)"

# 新采集格式（推荐）:
# 目录下应同时有:
#   rgb_{id}.mp4
#   depth_vis_{id}.mp4
#   depth_{id}/000000.png ...
#   params_{id}.json
#
# run_realr1_dino_sam.py 会优先使用 depth_{id}/ 原始深度目录，
# 自动从 params_{id}.json 读取 depth_scale_m。
python /projects/zaijia001/FoundationPose/run_realr1_dino_sam.py \
  --data_dir /projects/zaijia001/R1/hand/d_pour_low \
  --video_id 1 \
  --mesh_file /projects/_hdd/zaijia/R1/obj_mesh/bottle.obj \
  --output_dir /projects/zaijia001/R1/object_pose/d_pour_low_bottle_1 \
  --prompt bottle
