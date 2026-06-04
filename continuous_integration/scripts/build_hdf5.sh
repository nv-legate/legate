#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_SCRIPT_NAME='build_hdf5.sh'

base_dir="${1:-build/hdf5}"
mkdir -p "${base_dir}"

# The HDF5 package doesn't support building against build trees seamlessly.
# Build and install the library to a prefix that can then be found and used.
hdf5_version="1.14.6"

archive_file="hdf5-${hdf5_version}.tar.gz"
download_file="${base_dir}/${archive_file}"
source_dir="${base_dir}/source"
build_dir="${base_dir}/build"
install_dir="${base_dir}/install"

wget "https://github.com/HDFGroup/hdf5/releases/download/hdf5_${hdf5_version}/${archive_file}" -O "${download_file}"
mkdir -p "${source_dir}"
tar zxf "${download_file}" -C "${source_dir}" --strip-components=2

cmake_options=(
  CMAKE_INSTALL_PREFIX="$(realpath -Lms "${install_dir}")"
  HDF5_LIB_INFIX="-legate"
  BUILD_SHARED_LIBS=ON
  BUILD_STATIC_LIBS=OFF
  BUILD_TESTING=OFF
  HDF5_BUILD_EXAMPLES=OFF
  HDF5_BUILD_HL_LIB=OFF
  HDF5_BUILD_CPP_LIB=OFF
  HDF5_BUILD_TOOLS=OFF
  HDF5_BUILD_UTILS=OFF
  HDF5_ENABLE_ALL_WARNINGS=OFF
  HDF5_ENABLE_PARALLEL=OFF
)

cmake \
  -S "${source_dir}" \
  -B "${build_dir}" \
  "${cmake_options[@]/#/-D}"

cmake --build "${build_dir}"
cmake --build "${build_dir}" --target install
