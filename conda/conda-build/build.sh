#!/bin/bash

echo -e "\n\n--------------------- CONDA/CONDA-BUILD/BUILD.SH -----------------------\n"

set -xeo pipefail

# Rewrite conda's -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY to
#                 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH
CMAKE_ARGS="$(echo "$CMAKE_ARGS" | $SED -r "s@_INCLUDE=ONLY@_INCLUDE=BOTH@g")"

# Add our options to conda's CMAKE_ARGS
CMAKE_ARGS+="
--log-level=VERBOSE
-DBUILD_SHARED_LIBS=ON
-DBUILD_MARCH=x86-64
-DLegion_USE_OpenMP=${USE_OPENMP:-OFF}
-DLegion_USE_Python=ON
-DLegion_BUILD_JUPYTER=ON
-DLegion_Python_Version=$($PYTHON --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f3 --complement)"

if [ -z "$UPLOAD_ENABLED" ]; then
  CMAKE_ARGS+="
-Dlegate_core_BUILD_TESTS=ON
-Dlegate_core_BUILD_DOCS=ON
"
fi

# We rely on an environment variable to determine if we need to build cpu-only bits
if [ -z "$CPU_ONLY" ]; then
  CMAKE_ARGS+="
-DLegion_USE_CUDA=ON
-DLegion_CUDA_ARCH=all-major"
else
  CMAKE_ARGS+="
-DLegion_USE_CUDA=OFF"
fi

# We rely on an environment variable to determine if we need to make a debug build.
CMAKE_PRESET=release-gcc
if [ -n "$DEBUG_BUILD" ]; then
CMAKE_PRESET=$LEGATE_CORE_CMAKE_PRESET
fi
CMAKE_ARGS+="
--preset ${CMAKE_PRESET}
"

export CMAKE_GENERATOR=Ninja
export CUDAHOSTCXX=${CXX}
export OPENSSL_DIR="$PREFIX"

echo "Environment"
env

echo "Build starting on $(date)"

CUDAFLAGS="-ccbin ${CXX} -isystem ${PREFIX}/include -L${PREFIX}/lib"
export CUDAFLAGS

SKBUILD_BUILD_OPTIONS=-j$CPU_COUNT \
$PYTHON -m pip install             \
  --root / --prefix "$PREFIX" \
  --no-deps --no-build-isolation    \
  --upgrade                         \
  . -vv

# Install Legion's Python CFFI bindings
cmake \
    --install _skbuild/*/cmake-build/_deps/legion-build/bindings/python \
    --prefix "$PREFIX"

echo "Build ending on $(date)"

# Legion leaves an egg-info file which will confuse conda trying to pick up the information
# Remove it so the legate-core is the only egg-info file added
rm -rf $SP_DIR/legion*egg-info