#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -euo pipefail

export PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc --all --ignore=2)}
export CMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL}

if [[ "${CI:-false}" == "true" ]]; then
  echo "Installing extra system packages"
  dnf install -y gcc-toolset-11-libatomic-devel
  # Enable gcc-toolset-11 environment
  source /opt/rh/gcc-toolset-11/enable
  # Verify compiler version
  gcc --version
  g++ --version
fi

echo "PATH: ${PATH}"

if [[ "${LEGATE_DIR:-}" == "" ]]; then
  # If we are running in an action then GITHUB_WORKSPACE is set.
  if [[ "${GITHUB_WORKSPACE:-}" == "" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    LEGATE_DIR="$(python "${script_dir}"/../../scripts/get_legate_dir.py)"
  else
    # Simple path witin GitHub actions workflows.
    LEGATE_DIR="${GITHUB_WORKSPACE}"
  fi
  export LEGATE_DIR
fi
package_dir="${LEGATE_DIR}/scripts/build/python/legate"
package_name="legate"

echo "Installing build requirements"
python -m pip install -v --prefer-binary -r continuous_integration/requirements-build.txt

cd "${package_dir}"

echo "Building HDF5 and installing into prefix"
"${LEGATE_DIR}/continuous_integration/scripts/build_hdf5.sh"

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

echo "Building ${package_name}"
if [[ ! -d "prefix" ]]; then
  echo "No prefix, HDF5 may not have built where we thought!"
  exit 1
fi

SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$(pwd)/prefix;-DLegion_USE_CUDA:BOOL=ON;-DCMAKE_CUDA_ARCHITECTURES:STRING=all-major;-DBUILD_SHARED_LIBS:BOOL=ON"
export SKBUILD_CMAKE_ARGS
echo "SKBUILD_CMAKE_ARGS='${SKBUILD_CMAKE_ARGS}'"

python -m pip wheel \
  -w "${LEGATE_DIR}/dist" \
  -v \
  --no-deps \
  --disable-pip-version-check \
  .

echo "Show dist contents"
pwd
ls -lh "${LEGATE_DIR}/dist"

echo "Repairing the wheel"
mkdir -p "${LEGATE_DIR}/final-dist"
export LD_LIBRARY_PATH="${LEGATE_DIR}/scripts/build/python/legate/prefix/lib"
python -m auditwheel repair \
  --exclude libcuda.so* \
  --exclude libcudart.so.* \
  --exclude libnccl.so.*  \
  -w "${LEGATE_DIR}/final-dist" \
  "${LEGATE_DIR}"/dist/*.whl

echo "Wheel has been repaired. Contents:"
ls -lh "${LEGATE_DIR}/final-dist"
