#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/saxpy-build"

cmake_args=(--fresh)
if [[ "${CMAKE_ARGS:-}" != '' ]]; then
  cmake_args+=("${CMAKE_ARGS}")
fi

cmake -B "${build_dir}" -S "${script_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}"
"${build_dir}"/bin/saxpy
