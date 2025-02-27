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

# The HDF5 package doesn't support building against build trees seamlessly.
# Build and install the library to a prefix that can then be found and used.
hdf5_version="1.14.6"

wget "https://github.com/HDFGroup/hdf5/releases/download/hdf5_${hdf5_version}/hdf5-${hdf5_version}.tar.gz" -O hdf5.tgz
mkdir -p hdf5-build
tar zvxf hdf5.tgz -C hdf5-build --strip-components=2
cd hdf5-build

install_prefix="$(pwd)/../prefix"

cmake \
  -DBUILD_TESTING=OFF \
  -DEXAMPLES_EXTERNALLY_CONFIGURED=OFF \
  -DH5EX_BUILD_EXAMPLES=OFF \
  -DH5EX_BUILD_HL_LIB=OFF \
  -DH5EX_BUILD_TESTING=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF \
  -DHDF5_BUILD_HL_LIB=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DDBUILD_STATIC_LIBS=OFF \
  -DHDF5_BUILD_TOOLS=ON \
  -DHDF_BUILD_UTILS=ON \
  -DHDF5_ENABLE_ALL_WARNINGS=OFF \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -B build -S .
cmake --build build
cmake --build build --target install
