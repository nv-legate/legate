#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_SCRIPT_NAME="build_wheel_linux.bash"

# Enable sccache for faster builds but disable it for CUDA (#1884) issues
# with the realm CUDA kernel embedding.
source legate-configure-sccache
unset CMAKE_CUDA_COMPILER_LAUNCHER

export CMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL:=8}

if [[ "${CI:-false}" == "true" ]]; then
  rapids-logger "Installing extra system packages"
  rapids-retry dnf install -y gcc-toolset-14-libatomic-devel openmpi-devel mpich-devel
  # Enable gcc-toolset-11 environment
  source /opt/rh/gcc-toolset-14/enable
  # Verify compiler version
  gcc --version
  g++ --version
fi

rapids-logger "PATH: ${PATH}"

if [[ "${LEGATE_DIR:-}" == "" ]]; then
  # If we are running in an action then GITHUB_WORKSPACE is set.
  if [[ "${GITHUB_WORKSPACE:-}" == "" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    LEGATE_DIR="$(python "${script_dir}"/../../scripts/get_legate_dir.py)"
  else
    # Simple path within GitHub actions workflows.
    LEGATE_DIR="${GITHUB_WORKSPACE}"
  fi
  export LEGATE_DIR
fi
package_dir="${LEGATE_DIR}/scripts/build/python/legate"
package_name="legate"

rapids-logger "Installing build requirements"
rapids-pip-retry install -v --prefer-binary -r continuous_integration/requirements-build.txt

# Recreate the missing symlink and add in the cmake config for UCC.
sitepkgs=$(python -c 'import site; print(site.getsitepackages()[0], end="")')
ln -fs "${sitepkgs}"/nvidia/libcal/cu12/lib/libucc.so.1 "${sitepkgs}"/nvidia/libcal/cu12/lib/libucc.so
ln -fs "${sitepkgs}"/nvidia/nccl/lib/libnccl.so.2 "${sitepkgs}"/nvidia/nccl/lib/libnccl.so
if [[ ! -d "${sitepkgs}/nvidia/libcal/cu12/lib/cmake" ]]; then
  mkdir -p "${sitepkgs}/nvidia/libcal/cu12/lib/cmake"
  cp -r "${LEGATE_DIR}/continuous_integration/scripts/ucc-cmake-config" "${sitepkgs}/nvidia/libcal/cu12/lib/cmake/ucc"
fi

cd "${package_dir}"

rapids-logger "Building HDF5 and installing into prefix"
"${LEGATE_DIR}/continuous_integration/scripts/build_hdf5.sh"

# Build the wrappers and install into their prefix
MPI_WRAPPERS_DIR="${LEGATE_DIR}"/scripts/build/mpi_wrappers
cmake \
  -B "${LEGATE_DIR}/buildompi" \
  -S "${MPI_WRAPPERS_DIR}" \
  -DMPI_HOME=/usr/lib64/openmpi \
  -DLEGATE_WRAPPER_MPI_SUFFIX=ompi \
  -DCMAKE_INSTALL_PREFIX="${LEGATE_DIR}/wrapper-prefix"
cmake --build "${LEGATE_DIR}/buildompi"
cmake --install "${LEGATE_DIR}/buildompi"
cmake \
  -B "${LEGATE_DIR}/buildmpich" \
  -S "${MPI_WRAPPERS_DIR}" \
  -DMPI_HOME=/usr/lib64/mpich \
  -DLEGATE_WRAPPER_MPI_SUFFIX=mpich \
  -DCMAKE_INSTALL_PREFIX="${LEGATE_DIR}/wrapper-prefix"
cmake --build "${LEGATE_DIR}/buildmpich"
cmake --install "${LEGATE_DIR}/buildmpich"

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

rapids-logger "Building ${package_name}"
if [[ ! -d "prefix" ]]; then
  rapids-logger "No prefix, HDF5 may not have built where we thought!"
  exit 1
fi

# TODO(cryos): https://github.com/nv-legate/legate.internal/issues/1894
# Improve the use of CMAKE_PREFIX_PATH to find legate and cutensor once
# scikit-build supports it.
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$(pwd)/prefix;${sitepkgs}/libucx;${sitepkgs}/nvidia/libcal/cu12;${sitepkgs}/nvidia/nccl"
export CMAKE_ARGS
SKBUILD_CMAKE_ARGS="-DLEGATE_WRAPPER_DIR=${LEGATE_DIR}/wrapper-prefix"
export SKBUILD_CMAKE_ARGS
rapids-logger "SKBUILD_CMAKE_ARGS='${SKBUILD_CMAKE_ARGS}'"

sccache --zero-stats

python -m pip wheel \
  -w "${LEGATE_DIR}/dist" \
  -v \
  --no-deps \
  --disable-pip-version-check \
  .

sccache --show-adv-stats

rapids-logger "Show dist contents"
pwd
ls -lh "${LEGATE_DIR}/dist"

rapids-logger "Repairing the wheel"
mkdir -p "${LEGATE_DIR}/final-dist"
export LD_LIBRARY_PATH="${LEGATE_DIR}/scripts/build/python/legate/prefix/lib"
python -m auditwheel repair \
  --exclude libcrypto.so.* \
  --exclude libcuda.so.* \
  --exclude libcudart.so.* \
  --exclude libevent_core.so.* \
  --exclude libevent_pthreads-2.so.* \
  --exclude libhwloc.so.* \
  --exclude libmpi.so.* \
  --exclude libmpi_cxx.so.* \
  --exclude libmpicxx.so.* \
  --exclude libnccl.so.* \
  --exclude libopen-*.so.* \
  --exclude libucc.so.* \
  --exclude libucs.so.* \
  --exclude libuct.so.* \
  --exclude libucm.so.* \
  --exclude libucp.so.* \
  -w "${LEGATE_DIR}/final-dist" \
  "${LEGATE_DIR}"/dist/*.whl

rapids-logger "Wheel has been repaired. Contents:"
ls -lh "${LEGATE_DIR}/final-dist"
