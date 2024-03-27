#!/usr/bin/env bash

echo -e "\n\n--------------------- CONDA/CONDA-BUILD/BUILD.SH -----------------------\n"

set -xeo pipefail

# Rewrite conda's -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY to
#                 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH
CMAKE_ARGS="$(echo "$CMAKE_ARGS" | $SED -r "s@_INCLUDE=ONLY@_INCLUDE=BOTH@g")"

configure_args=()
if [[ "${USE_OPENMP:-OFF}" == "OFF" ]]; then
  configure_args+=(--with-openmp=0)
else
  configure_args+=(--with-openmp)
fi

if [ -z "$UPLOAD_ENABLED" ]; then
  configure_args+=(--with-tests)
  configure_args+=(--with-docs)
fi

# We rely on an environment variable to determine if we need to build cpu-only bits
if [ -z "$CPU_ONLY" ]; then
  configure_args+=(--with-cuda)
else
  configure_args+=(--with-cuda=0)
fi

# We rely on an environment variable to determine if we need to make a debug build.
if [ -n "$DEBUG_BUILD" ]; then
  configure_args+=(--build-type=debug)
else
  configure_args+=(--build-type=release)
fi

export CUDAHOSTCXX="${CXX}"
export OPENSSL_DIR="${PREFIX}"
export CUDAFLAGS="-ccbin ${CXX} -isystem ${PREFIX}/include -L${PREFIX}/lib"
export LEGATE_CORE_ARCH='arch-conda'

echo "Environment"
env

echo "Build starting on $(date)"
./configure \
  --LEGATE_CORE_ARCH="${LEGATE_CORE_ARCH}" \
  --with-python \
  "${configure_args[@]}"

SKBUILD_BUILD_OPTIONS=-j$CPU_COUNT \
$PYTHON -m pip install             \
  --root /                         \
  --no-deps                        \
  --prefix "${PREFIX}"             \
  --no-build-isolation             \
  --cache-dir "${PIP_CACHE_DIR}"   \
  --disable-pip-version-check      \
  . -vv

echo "Build ending on $(date)"

# Legion leaves an egg-info file which will confuse conda trying to pick up the information
# Remove it so the legate-core is the only egg-info file added
rm -rf "${SP_DIR}"/legion*egg-info
