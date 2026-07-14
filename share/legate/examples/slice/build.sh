#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds the example in this directory against the local legate build.
# The driver (build-and-run-all.sh) exports CMAKE_ARGS with the proper flags
# (-Dlegate_ROOT=... -DCMAKE_BUILD_TYPE=Debug); it can also be run standalone.

set -eou pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"

cmake_args=(--fresh)
if [[ "${CMAKE_ARGS:-}" != '' ]]; then
  read -ra args <<< "${CMAKE_ARGS}"
  cmake_args+=("${args[@]}")
fi

cmake -B "${build_dir}" -S "${script_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}"
