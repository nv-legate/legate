#!/usr/bin/env bash

echo -e "\n\n--------------------- CONDA/LEGATE-PROFILER/BUILD.SH (nsys) -----------------------\n"
set -xeo pipefail

# Set CUDA environment explicitly (important for cmake detection)
NVCC_PATH=$(command -v nvcc) || exit 1
export CUDACXX="${NVCC_PATH}"

# Needed for cargo
export CMAKE_CUDA_COMPILER=${CUDACXX}

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
cmake "${CUDA_ARGS}" -DCMAKE_INSTALL_PREFIX="${TMP_INSTALL}" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DLegion_BUILD_RUST_PROFILER=ON -DLegion_BUILD_EXAMPLES=ON -DLegion_USE_NVTX=ON ..
make -j 16
make install
# 'PREFIX' is set by conda-build at runtime
# shellcheck disable=SC2154
cp "${TMP_INSTALL}/bin/legion_prof" "${PREFIX}/bin/"

rm -rf "${TMP_INSTALL}"

echo "Done"
