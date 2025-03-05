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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"

${CMAKE} -E rm -rf "${build_dir}"
${CMAKE} -S "${script_dir}" -B "${build_dir}"
${CMAKE} --build "${build_dir}"

exe_path="${build_dir}/gen_h5_data"

if [[ ! -f "${exe_path}" ]]; then
  echo >&2 "Failed to build executable: '${exe_path}'. Aborting."
  exit 1
fi

generated_dir="${script_dir}/generated"
mkdir -p "${generated_dir}"
echo "====================================================="
echo "Built executable at ${exe_path}. To generate the HDF5 dataset please execute:"
echo ""
echo "  ${exe_path} -i '${generated_dir}/virt_data'"
