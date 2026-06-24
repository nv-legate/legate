#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/misalign-build"

cmake_args=()
if [[ "${CMAKE_ARGS:-}" != '' ]]; then
  read -ra args <<< "${CMAKE_ARGS}"
  cmake_args+=("${args[@]}")
fi

cmake -B "${build_dir}" -S "${script_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}"
