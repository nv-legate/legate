#!/usr/bin/env bash
#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================
set -eou pipefail

CMAKE="${CMAKE:-cmake}"

command -v "${CMAKE}" >/dev/null 2>&1 || {
  echo >&2 "CMake: '${CMAKE}' could not be found or is not executable. Aborting."
  exit 1
}

if [[ "${CMAKE_INSTALL_PREFIX:-}" != "" ]]; then
  prefix="${CMAKE_INSTALL_PREFIX}"
elif [[ "${PREFIX:-}" != "" ]]; then
  prefix="${PREFIX}"
elif [[ "${DESTDIR:-}" != "" ]]; then
  prefix="${DESTDIR}"
else
  prefix=""
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

declare -a cmake_configure_args
cmake_configure_args=(-S "${script_dir}" -B "${script_dir}/build")
[[ -n "${CMAKE_CONFIGURE_ARGS:-${CMAKE_ARGS:-}}" ]] && cmake_configure_args+=("${CMAKE_CONFIGURE_ARGS:-${CMAKE_ARGS:-}}")

declare -a cmake_build_args
cmake_build_args=(--build "${script_dir}/build")
[[ -n "${CMAKE_BUILD_ARGS:-}" ]] && cmake_build_args+=("${CMAKE_BUILD_ARGS}")

declare -a cmake_install_args
cmake_install_args=(--install "${script_dir}/build")
[[ -n "${CMAKE_INSTALL_ARGS:-}" ]] && cmake_build_args+=("${CMAKE_INSTALL_ARGS}")

if [[ "${prefix}" != "" ]]; then
  cmake_configure_args+=("-DCMAKE_INSTALL_PREFIX=${prefix}")
  # Export the same value as all 3
  export CMAKE_INSTALL_PREFIX="${prefix}"
  export DESTDIR="${prefix}"
  export PREFIX="${prefix}"
fi

${CMAKE} -E rm -rf "${script_dir}/build"
${CMAKE} "${cmake_configure_args[@]}"
${CMAKE} "${cmake_build_args[@]}"
${CMAKE} "${cmake_install_args[@]}"
