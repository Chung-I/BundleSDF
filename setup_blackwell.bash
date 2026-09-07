#!/bin/bash
# Blackwell (sm_120) setup for BundleSDF — replacement for setup.bash.
# Verified end to end on Ubuntu 24.04, RTX 5090, CUDA 12.8, gcc 13.3, no root.
#
# Why this exists. The original setup.bash:
#   1. sudo apt-get installs ~50 -dev packages (needs root),
#   2. builds Eigen, OpenCV + opencv_contrib, PCL, pybind11 and yaml-cpp from source
#      with `make -j$(nproc)` (hours of compiling, and a reliable OOM on a 30 GB box),
#   3. pins torch==2.4.0 and a CUDA 11.8 pytorch3d wheel, neither of which produces
#      sm_120 code, so nothing runs on an RTX 5090.
#
# Here all C++ dependencies but one come from conda-forge, and every CUDA component is
# built for sm_120. The exception is OpenCV: conda-forge's build ships *no* CUDA modules
# (0 cuda* headers, 0 libopencv_cuda* libraries) and BundleTrack needs cudafeatures2d,
# cudaimgproc and cudaoptflow, so OpenCV is still compiled from source.
#
# Prerequisites: micromamba on PATH; a CUDA >= 12.8 toolkit (12.8 is the first nvcc that
# emits sm_120) in $CUDA_ENV; NVIDIA driver >= 570.
#
# Usage:  bash setup_blackwell.bash
set -eo pipefail

BUNDLESDF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA="${MAMBA:-$HOME/.local/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

CUDA_ENV="${CUDA_ENV:-$MAMBA_ROOT_PREFIX/envs/r2s-cuda}"
DEPS_ENV="${DEPS_ENV:-$MAMBA_ROOT_PREFIX/envs/r2s-bundlesdf}"
OCV_PREFIX="$BUNDLESDF_DIR/local"          # where our CUDA OpenCV is installed
OCV_SRC="$BUNDLESDF_DIR/.opencv_build"

# ---------------------------------------------------------------------------
# 1. C++ dependencies (replaces the apt list AND four of the five source builds)
# ---------------------------------------------------------------------------
# pcl>=1.12 is required, not optional: conda's older pcl 1.8 includes
# boost/detail/endian.hpp, which Boost removed in 1.69.
if [ ! -d "$DEPS_ENV" ]; then
  "$MAMBA" create -y -n "$(basename "$DEPS_ENV")" -c conda-forge \
    eigen "pcl>=1.12" yaml-cpp pybind11 boost-cpp flann glog gflags \
    hdf5 proj protobuf zeromq cppzmq cmake ninja \
    libgl-devel libopengl-devel libglu freeglut glew openmpi mesalib
fi

# ---------------------------------------------------------------------------
# 2. Build environment
# ---------------------------------------------------------------------------
export CUDA_HOME="$CUDA_ENV"
# $CUDA_HOME/nvvm/bin must be on PATH: the deprecated FindCUDA module (used by
# BundleTrack) invokes nvcc such that it looks up its internal compiler `cicc` via PATH
# rather than relative to itself, giving "sh: 1: cicc: not found".
export PATH="$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$DEPS_ENV/bin:$PATH"

# torch's cpp_extension hardcodes -I$CUDA_HOME/include and -L$CUDA_HOME/lib64, but a
# conda cuda-toolkit keeps headers in targets/x86_64-linux/include and has no lib64
# (and $CUDA_HOME/include is already occupied by unrelated headers from other conda
# packages). Without these symlinks every CUDA extension fails with
#   fatal error: cuda_runtime_api.h: No such file or directory
ln -sfn "$CUDA_HOME"/targets/x86_64-linux/include/* "$CUDA_HOME/include/" 2>/dev/null || true
[ -d "$CUDA_HOME/lib64" ] || ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"

export CPATH="$DEPS_ENV/include/eigen3:$DEPS_ENV/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$DEPS_ENV/include/eigen3:$DEPS_ENV/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$DEPS_ENV/lib:$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export IGNORE_TORCH_VER=1
export OPENCV_IO_ENABLE_OPENEXR=1

# Bounded on purpose: parallel nvcc peaks near 1 GB per job, and -j$(nproc) has
# triggered the OOM killer on this machine.
export MAX_JOBS="${MAX_JOBS:-4}"

# Read by BundleTrack/CMakeLists.txt; its find_package(PCL ... NO_DEFAULT_PATH)
# otherwise searches only ../local.
export BUNDLESDF_DEPS_PREFIX="$DEPS_ENV"

cd "$BUNDLESDF_DIR"

# ---------------------------------------------------------------------------
# 3. Python environment
# ---------------------------------------------------------------------------
[ -d .venv ] || python3.10 -m venv .venv
PY="$BUNDLESDF_DIR/.venv/bin/python"

# pip must exist explicitly: a uv-created venv ships none, and several steps below
# shell out to `python -m pip`.
"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$PY" -m pip install --upgrade pip setuptools wheel ninja

# Everything below uses --no-build-isolation (mandatory: these packages import torch at
# build time, which an isolated build env lacks), so build backends must be present up
# front. Several dependencies fail to declare their own — gpustat needs setuptools_scm,
# fpsample needs scikit_build_core.
"$PY" -m pip install setuptools_scm scikit_build_core cmake pybind11 Cython numpy

# torch 2.3/2.4 have no sm_120 kernels; 2.7.1+cu128 is the first line that does.
"$PY" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# Upstream installed a prebuilt py310_cu118_pyt201 wheel; no cu128 wheel exists, so build
# from source. main rather than the V0.7.8 tag, which predates torch 2.7.
"$PY" -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@main"

# Upstream pinned numpy==1.26.1, Cython==0.29.20, scikit-image==0.17.2, networkx==2.2 —
# ~2020 releases that no longer build on cp310 with modern setuptools, and vestigial
# anyway: setup.bash reinstalls scikit-image unpinned and upgrades networkx at the end.
"$PY" -m pip install \
  trimesh opencv-python wandb matplotlib imageio tqdm open3d ruamel.yaml sacred \
  kornia pymongo scipy scikit-image networkx transformations einops gputil xatlas \
  rtree pytinyrenderer chardet openpyxl pyrender PyOpenGL-accelerate \
  dearpygui pymeshlab yacs
"$PY" -c "import imageio; imageio.plugins.freeimage.download()" || true

# LoFTR weights: loftr_wrapper.py loads BundleTrack/LoFTR/weights/outdoor_ds.ckpt at
# runtime and the directory ships empty.
if [ ! -f "$BUNDLESDF_DIR/BundleTrack/LoFTR/weights/outdoor_ds.ckpt" ]; then
  "$PY" -m pip install gdown
  mkdir -p "$BUNDLESDF_DIR/BundleTrack/LoFTR/weights"
  "$PY" -m gdown --folder \
    "https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp" \
    -O "$BUNDLESDF_DIR/BundleTrack/LoFTR/weights"
fi

# ---------------------------------------------------------------------------
# 4. OpenCV 4.12.0 + contrib, with CUDA  (the one unavoidable source build)
# ---------------------------------------------------------------------------
# 4.12.0 specifically: opencv_contrib dropped rgbd and xfeatures2d after it, and
# BundleTrack includes opencv2/rgbd.hpp and opencv2/xfeatures2d/nonfree.hpp. 4.12 is also
# new enough to accept CUDA_ARCH_BIN=12.0.
# Only the 17 modules BundleTrack actually references are built; python bindings are off
# (nothing on the Python side uses cv2.cuda — pip's opencv-python covers that).
if [ ! -f "$OCV_PREFIX/lib/libopencv_cudafeatures2d.so" ]; then
  mkdir -p "$OCV_SRC" && cd "$OCV_SRC"
  [ -d opencv ]         || git clone -q --depth 1 --branch 4.12.0 https://github.com/opencv/opencv.git
  [ -d opencv_contrib ] || git clone -q --depth 1 --branch 4.12.0 https://github.com/opencv/opencv_contrib.git
  rm -rf opencv/build && mkdir -p opencv/build && cd opencv/build
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$OCV_PREFIX" \
    -DOPENCV_EXTRA_MODULES_PATH="$OCV_SRC/opencv_contrib/modules" \
    -DBUILD_LIST=core,imgproc,imgcodecs,highgui,videoio,calib3d,features2d,flann,video,cudev,cudaarithm,cudawarping,cudaimgproc,cudafeatures2d,cudaoptflow,xfeatures2d,rgbd \
    -DWITH_CUDA=ON -DCUDA_ARCH_BIN=12.0 -DCUDA_ARCH_PTX= -DCUDA_FAST_MATH=ON -DWITH_CUDNN=OFF \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
    -DBUILD_opencv_apps=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
    -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_VTK=OFF -DWITH_OPENGL=OFF -DWITH_IPP=OFF \
    -DWITH_VA=OFF -DWITH_VA_INTEL=OFF -DWITH_LIBVA=OFF \
    -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=ON
  # WITH_VA=OFF matters: OpenCV finds VA-API *headers* in the conda env (it is on PATH
  # for cmake) but the libraries are not linkable, and opencv_core fails at link with
  # "/usr/bin/ld: cannot find -lva".
  make -j"$MAX_JOBS"
  make install
fi

# ---------------------------------------------------------------------------
# 5. CUDA python components (all for sm_120)
# ---------------------------------------------------------------------------
# kaolin is NOT optional despite Utils.py wrapping `import kaolin` in try/except:
# OctreeManager (kaolin.ops.spc) is constructed in nerf_runner.py, the reconstruction
# path run_asset_generation.py invokes. All 13 kaolin APIs used still exist in 0.18.0.
"$PY" -m pip install --no-build-isolation \
  "git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.18.0"

# BundleSDF's own extension. mycuda/setup.py had -arch=sm_86 hardcoded; it is sm_120 now.
# NOTE: editable (-e) is required, not a preference. setup.py declares top-level
# extensions named "common"/"gridencoder", but Utils.py does `from mycuda import common`.
# A non-editable install puts common.so in site-packages as a *top-level* module and
# leaves mycuda/ without it, so that import fails with
#   ImportError: cannot import name 'common' from 'mycuda' (unknown location)
# and a NeRF worker process then dies, leaving the parent blocked on its pipe forever.
# -e builds the .so in place inside mycuda/, which makes it a real submodule.
cd "$BUNDLESDF_DIR/mycuda"
rm -rf build ./*egg*
"$PY" -m pip install --no-build-isolation -e .

# ---------------------------------------------------------------------------
# 6. BundleTrack (C++)
# ---------------------------------------------------------------------------
# CMakeLists.txt had CMAKE_CUDA_ARCHITECTURES "52 60 61 70 75 80 86" and a matching
# gencode list; both target 120 now. OpenCV_DIR is passed explicitly so it picks our
# CUDA build rather than any conda OpenCV on the prefix path.
cd "$BUNDLESDF_DIR/BundleTrack"
rm -rf build && mkdir -p build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH="$OCV_PREFIX;$DEPS_ENV" \
  -DOpenCV_DIR="$OCV_PREFIX/lib/cmake/opencv4" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DOpenGL_GL_PREFERENCE=GLVND \
  -DPYTHON_EXECUTABLE="$PY"
make -j"$MAX_JOBS"

# Bake the library paths into the built objects. Without this, my_cpp.so leaves 21
# libraries unresolved (OpenCV, PCL, yaml-cpp, ...) and would need LD_LIBRARY_PATH set --
# which run_asset_generation.py's subprocesses would not reliably inherit.
PATCHELF="$DEPS_ENV/bin/patchelf"
[ -x "$PATCHELF" ] || "$MAMBA" install -y -q -n "$(basename "$DEPS_ENV")" -c conda-forge patchelf
PYLIB="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
RPATH="$OCV_PREFIX/lib:$DEPS_ENV/lib:$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:$PYLIB:$PWD"
for f in my_cpp*.so libBundleTrack.so libMY_CUDA_LIB.so; do
  [ -f "$f" ] && "$PATCHELF" --set-rpath "$RPATH" "$f"
done

cat <<EOF

BundleSDF setup complete.
  python : $PY
  deps   : $DEPS_ENV
  opencv : $OCV_PREFIX  (CUDA build)
  cuda   : $CUDA_HOME

Library paths are baked into the .so files, but one preload is still required:
  export LD_PRELOAD=$DEPS_ENV/lib/libjpeg.so.8
  export PYTHONPATH=$BUNDLESDF_DIR:$BUNDLESDF_DIR/BundleTrack/build:\$PYTHONPATH

torchvision bundles its own libjpeg whose SONAME is also libjpeg.so.8 but which lacks
the jpeg12_* symbols. bundlesdf.py imports torch/torchvision before my_cpp, so the
loader reuses torchvision's copy and conda's libtiff.so.6 then fails with
"undefined symbol: jpeg12_write_raw_data". Preloading conda's libjpeg fixes it.
EOF
