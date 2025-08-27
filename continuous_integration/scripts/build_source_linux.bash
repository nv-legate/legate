#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_SCRIPT_NAME="build_source_linux.bash"

# Enable sccache for faster builds but disable it for CUDA (#1884) issues
# with the realm CUDA kernel embedding.
source legate-configure-sccache
unset CMAKE_CUDA_COMPILER_LAUNCHER

export CMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL:=16}

GCC_VER=${GCC_VER:-10}

if [[ "${CI:-false}" == "true" ]]; then
  rapids-logger "Installing extra system packages"
  dnf install -y gcc-toolset-"${GCC_VER}" gcc-toolset-"${GCC_VER}"-libatomic-devel \
      git curl python3.11 python3.11-pip
  # Enable gcc-toolset environment, MANPATH is unbound
  set +u
  source /opt/rh/gcc-toolset-"${GCC_VER}"/enable
  set -u
  # Verify compiler version
  gcc --version
  g++ --version

  # Quick and hackish way to get a new cmake version and ninja.
  python3 -m venv ~/env
  source ~/env/bin/activate
  pip install cmake ninja

  # Install sccache to improve compilation speeds.
  export SCCACHE_VER=${SCCACHE_VER:-0.10.0}
  mkdir -p dl
  curl -o dl/sccache.tgz -L "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VER}/sccache-v${SCCACHE_VER}-x86_64-unknown-linux-musl.tar.gz"
  tar -C dl -xf dl/sccache.tgz
  mv "dl/sccache-v${SCCACHE_VER}-x86_64-unknown-linux-musl/sccache" /usr/bin/sccache
  chmod +x /usr/bin/sccache
fi

rapids-logger "PATH: ${PATH}"

if [[ "${LEGATE_DIR:-}" == "" ]]; then
  # If we are running in an action then GITHUB_WORKSPACE is set.
  if [[ "${GITHUB_WORKSPACE:-}" == "" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    LEGATE_DIR="$(python "${script_dir}"/../../scripts/get_legate_dir.py)"
  else
    # Simple path within GitHub actions workflows.
    LEGATE_DIR="${GITHUB_WORKSPACE}"
  fi
  export LEGATE_DIR
fi

sccache --zero-stats
rapids-logger "Building the main project"
cmake -G Ninja -B build -S src -Dlegate_USE_CUDA=ON -Dlegate_BUILD_TESTS=ON
cmake --build build -v
sccache --show-adv-stats
