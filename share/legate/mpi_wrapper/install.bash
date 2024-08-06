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

DIRNAME="${DIRNAME:-dirname}"
READLINK="${READLINK:-readlink}"

script_dir="$(${DIRNAME} "$(${READLINK} -f "${BASH_SOURCE[0]}")")"

RM="${RM:-rm}"
CMAKE="${CMAKE:-cmake}"

if [[ "${CMAKE_INSTALL_PREFIX:-}" != "" ]]; then
  prefix="${CMAKE_INSTALL_PREFIX}"
elif [[ "${PREFIX:-}" != "" ]]; then
  prefix="${PREFIX}"
elif [[ "${DESTDIR:-}" != "" ]]; then
  prefix="${DESTDIR}"
else
  prefix=""
fi

cmake_args=()
if [[ "${prefix}" != "" ]]; then
  cmake_args+=("-DCMAKE_INSTALL_PREFIX=${prefix}")
  # Export the same value as all 3
  export CMAKE_INSTALL_REFIX="${prefix}"
  export DESTDIR="${prefix}"
  export PREFIX="${prefix}"
fi

${RM} -rf "${script_dir}/build" && \
  ${CMAKE} -S "${script_dir}" -B "${script_dir}/build" "${cmake_args[@]}" && \
  ${CMAKE} --build "${script_dir}/build" && \
  ${CMAKE} --install "${scipt_dir}/build"
