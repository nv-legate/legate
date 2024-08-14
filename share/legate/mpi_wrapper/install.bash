#!/usr/bin/env bash
#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================
set -xeou pipefail

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

declare -a cmake_configure_args
cmake_configure_args=("${CMAKE_CONFIGURE_ARGS:-${CMAKE_ARGS:-}}")

if [[ "${prefix}" != "" ]]; then
  cmake_configure_args+=("-DCMAKE_INSTALL_PREFIX=${prefix}")
  # Export the same value as all 3
  export CMAKE_INSTALL_PREFIX="${prefix}"
  export DESTDIR="${prefix}"
  export PREFIX="${prefix}"
fi

declare -a cmake_build_args
cmake_build_args=("${CMAKE_BUILD_ARGS:-}")

declare -a cmake_install_args
cmake_install_args=("${CMAKE_INSTALL_ARGS:-}")

DIRNAME="${DIRNAME:-dirname}"
READLINK="${READLINK:-readlink}"

script_dir="$(${DIRNAME} "$(${READLINK} -f "${BASH_SOURCE[0]}")")"

${CMAKE} -E rm -rf "${script_dir}/build" && \
  ${CMAKE} -S "${script_dir}" -B "${script_dir}/build" "${cmake_configure_args[@]}" && \
  ${CMAKE} --build "${script_dir}/build" "${cmake_build_args[@]}" && \
  ${CMAKE} --install "${scipt_dir}/build" "${cmake_install_args[@]}"
