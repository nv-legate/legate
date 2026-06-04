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
export CUDA_MAJOR_VER=${CUDA_MAJOR_VER:=13}

if [[ "${CI:-false}" == "true" ]]; then
  rapids-logger "Installing extra system packages"
  rapids-retry dnf install -y gcc-toolset-14-libatomic-devel
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
package_suffix=""
if [[ "${CUDA_MAJOR_VER}" == "12" ]]; then
  package_suffix="-cu12"
fi
package_dir="${LEGATE_DIR}/scripts/build/python/legate${package_suffix}"
package_name="legate${package_suffix}"
cd "${LEGATE_DIR}"

rapids-logger "Installing build requirements"
rapids-pip-retry install -v --prefer-binary \
  -r "continuous_integration/requirements-build-common.txt" \
  -r "continuous_integration/requirements-build${package_suffix}.txt"

hdf5_base_dir="build/hdf5"
hdf5_install_dir="${hdf5_base_dir}/install"
hdf5_install_cmake="${hdf5_install_dir}/cmake/hdf5-config.cmake"
if [[ -f "${hdf5_install_cmake}" ]]; then
  rapids-logger "Using existing HDF5 install in ${hdf5_install_dir}"
else
  rapids-logger "Building HDF5 and installing into ${hdf5_install_dir}"
  "continuous_integration/scripts/build_hdf5.sh" "${hdf5_base_dir}"
  if [[ ! -f "${hdf5_install_cmake}" ]]; then
    rapids-logger "No ${hdf5_install_cmake}, HDF5 may not have built where we thought!"
    exit 1
  fi
fi

rapids-logger "Building ${package_name}"

site_packages_dir="$(python -c 'import site; print(site.getsitepackages()[0])')"
pip_nvidia_dir="${site_packages_dir}/nvidia"
nccl_dir="${pip_nvidia_dir}/nccl"
nccl_lib="${nccl_dir}/lib/libnccl.so.2"

if [[ "${CUDA_MAJOR_VER}" == "12" ]]; then
  pip_nvidia_cmake="${pip_nvidia_dir}/cufile"
else
  pip_nvidia_cmake="${pip_nvidia_dir}"
fi

cmake_defines=(
  CMAKE_PREFIX_PATH:STRING="${pip_nvidia_cmake}"
  legate_USE_HDF5:BOOL=ON
  legate_USE_HDF5_VFD_GDS:BOOL=ON
  HDF5_DIR:PATH="$(dirname "${LEGATE_DIR}/${hdf5_install_cmake}")"
  NCCL_ROOT:PATH="${nccl_dir}"
  NCCL_LIBRARY:FILEPATH="${nccl_lib}"
)
rapids-logger "cmake_defines=(" "${cmake_defines[@]/#/  }" ")"

sccache --zero-stats

# build with '--no-build-isolation', for better sccache hit rate
python -m pip wheel \
  -w "${LEGATE_DIR}/dist" \
  -v \
  --no-build-isolation \
  --no-deps \
  --disable-pip-version-check \
  "${cmake_defines[@]/#/--config-settings=cmake.define.}" \
  "${package_dir}"

sccache --show-adv-stats

rapids-logger "Show dist contents"
pwd
ls -lh "${LEGATE_DIR}/dist"

rapids-logger "Repairing the wheel"
mkdir -p "${LEGATE_DIR}/final-dist"

# Needed to help auditwheel find the HDF5 library we built.
export AUDITWHEEL_LD_LIBRARY_PATH="${LEGATE_DIR}/${hdf5_install_dir}/lib"

auditwheel_excludes=(
  libcrypto.so.*
  libcuda.so.*
  libcufile.so.*
  libevent*.so.*
  libhwloc.so.*
  libmpi*.so.*
  libnccl.so.*
  libopen-*.so.*
  libuc?.so.*
)

# whitespace splitting is intentional here
# shellcheck disable=SC2048,SC2086
python -m auditwheel repair \
  -w "${LEGATE_DIR}/final-dist" \
  ${auditwheel_excludes[*]/#/--exclude } \
  "${LEGATE_DIR}"/dist/*.whl

rapids-logger "Wheel has been repaired. Contents:"
ls -lh "${LEGATE_DIR}/final-dist"
