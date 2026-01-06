#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo -e "\n\n--------------------- CONDA/LEGATE-PROFILER/BUILD.SH (nsys) -----------------------\n"
set -xeo pipefail

# Set CUDA environment explicitly (important for cmake detection)
if [[ "${BUILD_WITH_CUDA:-0}" == "1" ]]; then
    NVCC_PATH=$(command -v nvcc) || exit 1
    export CUDACXX="${NVCC_PATH}"
fi

# Needed for cargo
export CMAKE_CUDA_COMPILER=${CUDACXX}

# Ensure all toolchains (C/C++ and Rust) use the conda sysroot and linker so we
# don't pick up host glibc headers/libraries.
if [[ -n "${CONDA_BUILD_SYSROOT:-}" ]]; then
    export CFLAGS="${CFLAGS:+${CFLAGS} }--sysroot=${CONDA_BUILD_SYSROOT}"
    export CXXFLAGS="${CXXFLAGS:+${CXXFLAGS} }--sysroot=${CONDA_BUILD_SYSROOT}"
    export CPPFLAGS="${CPPFLAGS:+${CPPFLAGS} }--sysroot=${CONDA_BUILD_SYSROOT}"
    export LDFLAGS="${LDFLAGS:+${LDFLAGS} }--sysroot=${CONDA_BUILD_SYSROOT}"

    rust_linker="${CC:-cc}"
    export RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }-Clinker=${rust_linker} -C link-arg=--sysroot=${CONDA_BUILD_SYSROOT}"
    case "$(gcc -dumpmachine)" in
        x86_64-*)  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="${rust_linker}" ;;
        aarch64-*) export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="${rust_linker}" ;;
        *)         echo "Unsupported build triple for cargo linker override" >&2; exit 1 ;;
    esac
fi

# Build the legate profiler
# shellcheck disable=SC2154
GIT_COMMIT=$(git -C "${SRC_DIR}" rev-parse HEAD)
echo "Legion checked-out with commit: ${GIT_COMMIT}"

BUILD_DIR="${SRC_DIR}/build"
TMP_INSTALL="${SRC_DIR}/tmp_install"
mkdir -p "${BUILD_DIR}" "${TMP_INSTALL}"

if [[ "${BUILD_WITH_CUDA:-0}" == '1' ]]; then
    CUDA_ARGS="-DLegion_USE_CUDA=ON -DCMAKE_CUDA_COMPILER=${CUDACXX}"
else
    CUDA_ARGS="-DLegion_USE_CUDA=OFF"
fi

cd "${BUILD_DIR}"
export CMAKE_INSTALL_PREFIX="${TMP_INSTALL}"
cmake "${CUDA_ARGS}" -DCMAKE_INSTALL_PREFIX="${TMP_INSTALL}" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DLegion_BUILD_RUST_PROFILER=ON -DLegion_BUILD_EXAMPLES=ON -DLegion_USE_NVTX=ON -DLegion_HIJACK_CUDART=OFF ..
make -j 16
make install
# 'PREFIX' is set by conda-build at runtime
# shellcheck disable=SC2154
cp "${TMP_INSTALL}/bin/legion_prof" "${PREFIX}/bin/legate_prof"

rm -rf "${TMP_INSTALL}"

echo "Done"
