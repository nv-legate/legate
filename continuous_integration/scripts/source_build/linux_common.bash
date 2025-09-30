#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_SCRIPT_NAME="linux_common.bash"

# Enable sccache for faster builds but disable it for CUDA (#1884) issues
# with the realm CUDA kernel embedding.
source legate-configure-sccache
unset CMAKE_CUDA_COMPILER_LAUNCHER

source pretty_printing.bash

export CMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL:=16}

if [[ "${CI:-false}" == "true" ]]; then
  GCC_VER=${GCC_VER:-10}
  # Enable gcc-toolset environment, MANPATH is unbound. We cannot do this in the github
  # action because this sets a bunch of environment variables that don't carry over.
  set +u
  # Silence shellcheck, we know this thing exists
  # shellcheck source=/dev/null
  source /opt/rh/gcc-toolset-"${GCC_VER}"/enable
  set -u

  # Verify compiler version
  gcc --version
  g++ --version
fi

if [[ "${LEGATE_DIR:-}" == "" ]]; then
  # If we are running in an action then GITHUB_WORKSPACE is set.
  if [[ "${GITHUB_WORKSPACE:-}" == "" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    LEGATE_DIR="$(python "${script_dir}"/../../../scripts/get_legate_dir.py)"
  else
    # Simple path within GitHub actions workflows.
    LEGATE_DIR="${GITHUB_WORKSPACE}"
  fi
  export LEGATE_DIR
fi

export CMAKE_PRESET=${CMAKE_PRESET:-ci_release_cuda_tests}

run_command "Environment" env
# Need to cd into src, since that contains CMakePresets.json, because cmake requires it to
# exist at cwd. There is currently no option to tell it to source the presets from another
# directory.
cd "${LEGATE_DIR}/src"
run_command "CMake Configure" cmake \
            -G Ninja \
            -B "${LEGATE_DIR}/build" \
            -S . \
            --preset "${CMAKE_PRESET}" \
            --log-context \
            --log-level=DEBUG

run_command "CMake Build" cmake \
            --build "${LEGATE_DIR}/build" \
            -v
