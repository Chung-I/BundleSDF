#!/bin/bash
# Blackwell (sm_120) setup for BundleSDF — replacement for setup.bash.
#
# Why this exists. The original setup.bash:
#   1. sudo apt-get installs ~50 -dev packages (needs root),
#   2. builds Eigen, OpenCV + opencv_contrib, PCL, pybind11 and yaml-cpp from source
#      with `make -j$(nproc)` (hours of compiling, and a reliable OOM on a 30 GB box),
#   3. pins torch==2.4.0 and a CUDA 11.8 pytorch3d wheel, neither of which produces
#      sm_120 code, so nothing runs on an RTX 5090.
#
# This script needs no root: the C++ dependencies come from a conda-forge environment,
# and every CUDA component is built for sm_120.
#
# Prerequisites:
#   - micromamba (or mamba/conda) on PATH
#   - a CUDA >= 12.8 toolkit; CUDA 12.8 is the first release whose nvcc emits sm_120
#   - NVIDIA driver >= 570
#
# Usage:  bash setup_blackwell.bash
set -eo pipefail

BUNDLESDF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA="${MAMBA:-$HOME/.local/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

CUDA_ENV="${CUDA_ENV:-$MAMBA_ROOT_PREFIX/envs/r2s-cuda}"
DEPS_ENV="${DEPS_ENV:-$MAMBA_ROOT_PREFIX/envs/r2s-bundlesdf}"

# ---------------------------------------------------------------------------
# 1. C++ dependencies (replaces the apt list AND the four from-source builds)
# ---------------------------------------------------------------------------
if [ ! -d "$DEPS_ENV" ]; then
  "$MAMBA" create -y -n r2s-bundlesdf -c conda-forge \
    eigen opencv pcl yaml-cpp pybind11 boost-cpp flann glog gflags \
    hdf5 proj protobuf zeromq cmake ninja
fi

# ---------------------------------------------------------------------------
# 2. Build environment
# ---------------------------------------------------------------------------
export CUDA_HOME="$CUDA_ENV"
export PATH="$CUDA_HOME/bin:$PATH"

# torch's cpp_extension hardcodes -I$CUDA_HOME/include and -L$CUDA_HOME/lib64, but a
# conda cuda-toolkit keeps its headers in targets/x86_64-linux/include and has no lib64
# (and $CUDA_HOME/include is already occupied by unrelated headers from other conda
# packages). Without these symlinks every CUDA extension fails with
#   fatal error: cuda_runtime_api.h: No such file or directory
ln -sfn "$CUDA_HOME"/targets/x86_64-linux/include/* "$CUDA_HOME/include/" 2>/dev/null || true
[ -d "$CUDA_HOME/lib64" ] || ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"

export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export IGNORE_TORCH_VER=1
export OPENCV_IO_ENABLE_OPENEXR=1

# Bounded on purpose: parallel nvcc peaks around 1 GB per job, and an unbounded
# -j$(nproc) has previously triggered the OOM killer on this machine.
export MAX_JOBS="${MAX_JOBS:-4}"

# BundleTrack/CMakeLists.txt reads this to find the conda-forge dependencies; its
# find_package(PCL ... NO_DEFAULT_PATH) otherwise searches only ../local.
export BUNDLESDF_DEPS_PREFIX="$DEPS_ENV"

cd "$BUNDLESDF_DIR"

# ---------------------------------------------------------------------------
# 3. Python environment
# ---------------------------------------------------------------------------
if [ ! -d .venv ]; then
  python3.10 -m venv .venv
fi
PY="$BUNDLESDF_DIR/.venv/bin/python"

# pip is required explicitly: a uv-created venv does not ship one, and several build
# steps shell out to `python -m pip`.
"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$PY" -m pip install --upgrade pip setuptools wheel ninja

# Build backends must be present up front because everything below uses
# --no-build-isolation (required: these packages import torch at build time, which an
# isolated build env does not have). Several dependencies do not declare their own
# backends — gpustat needs setuptools_scm, fpsample needs scikit_build_core.
"$PY" -m pip install setuptools_scm scikit_build_core cmake pybind11 Cython numpy

# torch 2.3/2.4 have no sm_120 kernels; 2.7.1+cu128 is the first line that does.
"$PY" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# Upstream installed a prebuilt py310_cu118_pyt201 wheel. No prebuilt wheel exists for
# cu128, so pytorch3d is built from source. main is used rather than the V0.7.8 tag,
# which predates torch 2.7 and trips over removed ATen APIs.
"$PY" -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@main"

# Upstream pinned numpy==1.26.1, Cython==0.29.20, scikit-image==0.17.2 and
# networkx==2.2. Those are ~2020 releases that no longer build on cp310 with modern
# setuptools, and they are vestigial anyway: setup.bash reinstalls scikit-image
# unpinned and runs `pip install --upgrade networkx` at the end, overwriting them.
"$PY" -m pip install \
  trimesh opencv-python wandb matplotlib imageio tqdm open3d ruamel.yaml sacred \
  kornia pymongo scipy scikit-image networkx transformations einops gputil xatlas \
  rtree pytinyrenderer chardet openpyxl pyrender PyOpenGL-accelerate

"$PY" -c "import imageio; imageio.plugins.freeimage.download()" || true

# ---------------------------------------------------------------------------
# 4. CUDA components (all built for sm_120)
# ---------------------------------------------------------------------------
# kaolin is NOT optional despite Utils.py wrapping `import kaolin` in try/except:
# OctreeManager (kaolin.ops.spc) is constructed in nerf_runner.py, which is the
# reconstruction path run_asset_generation.py invokes.
"$PY" -m pip install --no-build-isolation \
  "git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.18.0"

# BundleSDF's own extension. mycuda/setup.py had -arch=sm_86 hardcoded; it is sm_120 here.
cd "$BUNDLESDF_DIR/mycuda"
rm -rf build ./*egg*
"$PY" -m pip install --no-build-isolation .

# ---------------------------------------------------------------------------
# 5. BundleTrack (C++)
# ---------------------------------------------------------------------------
# CMakeLists.txt had CMAKE_CUDA_ARCHITECTURES "52 60 61 70 75 80 86" and a matching
# gencode list; both now target 120. Without that it links fine and dies at kernel launch.
cd "$BUNDLESDF_DIR/BundleTrack"
rm -rf build && mkdir -p build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH="$DEPS_ENV" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPYTHON_EXECUTABLE="$PY"
make -j"$MAX_JOBS"

echo
echo "BundleSDF setup complete."
echo "  python : $PY"
echo "  deps   : $DEPS_ENV"
echo "  cuda   : $CUDA_HOME"
