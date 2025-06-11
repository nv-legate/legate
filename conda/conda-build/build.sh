#!/usr/bin/env bash

echo -e "\n\n--------------------- CONDA/CONDA-BUILD/BUILD.SH -----------------------\n"

set -xeo pipefail

# LICENSE, README.md, conda/, and configure are guaranteed to always be at the root
# directory. If we can't find them, then probably we are not in the root directory.
if [[ ! -f LICENSE ]] || [[ ! -f README.md ]] || [[ ! -d conda ]] || [[ ! -f configure ]]; then
  echo "Must run this script from the root directory"
  exit 1
fi

# If run through CI, BUILD_MARCH is set externally. If it is not set, try to set it.
ARCH=$(uname -m)
if [[ -z "${BUILD_MARCH}" ]]; then
    if [[ "${ARCH}" = "aarch64" ]]; then
        # Use the gcc march value used by aarch64 Ubuntu.
        BUILD_MARCH=armv8-a
    else
        # Use uname -m otherwise
        BUILD_MARCH=$(uname -m | tr '_' '-')
    fi
fi

. continuous_integration/scripts/tools/pretty_printing.bash

if [[ "${LEGATE_CI:-0}" == '0' ]]; then
  # not running in CI, define a dummy version of this function
  function run_command()
  {
    { set +x; } 2>/dev/null;

    shift # ignore group name argument
    local command=("$@")

    "${command[@]}"
  }
else
  export LEGATE_CI_GROUP=0
  export PYTHONUNBUFFERED=1
fi

function preamble()
{
  set -xeo pipefail
  # Rewrite conda's -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY to
  #                 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH
  CMAKE_ARGS="${CMAKE_ARGS//_INCLUDE=ONLY/_INCLUDE=BOTH}"
  # Conda sets these to -I/some/path/in/cuda/toolkit, but this breaks all sorts of stuff
  # in CMake. Since we use CCCL as a third-party dependency, we add its headers with
  # -isystem, but since conda adds additional headers with -I, those take precedence over
  # -isystem, meaning they are effectively shadowed.
  #
  # So thanks, conda, once again, for doing something "helpful"!
  unset NVCC_APPEND_FLAGS
  unset NVCC_PREPEND_FLAGS

  configure_args=()
  if [[ "${USE_OPENMP:-OFF}" == 'OFF' ]]; then
    configure_args+=(--with-openmp=0)
  else
    configure_args+=(--with-openmp)
  fi

  if [[ "${BUILD_TESTS:-0}" == '1' ]]; then
    configure_args+=(--with-tests)
    configure_args+=(--with-benchmarks)
  fi

  # We rely on an environment variable to determine if we need to build cpu-only bits
  if [[ "${CPU_ONLY:-0}" == '0' ]]; then
    configure_args+=(--with-cuda)
    configure_args+=(--with-cal)
  else
    configure_args+=(--with-cuda=0)
  fi

  configure_args+=(--build-type="${LEGATE_BUILD_MODE}")

# shellcheck disable=SC2154
case "${LEGATE_NETWORK}" in
  "ucx")
    configure_args+=(--with-ucx)
    ;;
  "gex")
    configure_args+=(--with-gasnet)
    configure_args+=(--)
    configure_args+=(-DLegion_USE_GASNETEX_WRAPPER=ON)
    ;;
  *)
    echo "${LEGATE_NETWORK} is not a valid choice for the network interface"
    exit 1
    ;;
esac

  # ${CXX} is set by conda compiler package. Disable shellcheck warning.
  # shellcheck disable=SC2154
  export CUDAHOSTCXX="${CXX}"
  # ${PREFIX} is set by conda build. Ignore shellcheck warning.
  # shellcheck disable=SC2154
  export OPENSSL_DIR="${PREFIX}"
  # shellcheck disable=SC2154
  LEGATE_DIR="$(${PYTHON} ./scripts/get_legate_dir.py)"
  export LEGATE_DIR
  export LEGATE_ARCH='arch-conda'

  # In classic conda fashion, it sets a bunch of environment variables for you but as
  # usual this just ends up creating more headaches. We don't want FORTIFY_SOURCE because
  # GCC and clang error with:
  #
  # /tmp/conda-croot/legate/_build_env/x86_64-conda-linux-gnu/sysroot/usr/include/features.h:330:4:
  # error: #warning _FORTIFY_SOURCE requires compiling with optimization (-O)
  # [-Werror=cpp]
  #     330 | #  warning _FORTIFY_SOURCE requires compiling with optimization (-O)
  #         |    ^~~~~~~
  #
  # Thanks conda, such a great help!
  if [[ ${LEGATE_BUILD_MODE} == *debug* ]]; then
    CPPFLAGS="${CPPFLAGS//-D_FORTIFY_SOURCE=[0-9]/}"
    export CPPFLAGS
    DEBUG_CPPFLAGS="${DEBUG_CPPFLAGS//-D_FORTIFY_SOURCE=[0-9]/}"
    export DEBUG_CPPFLAGS
    CFLAGS="${CFLAGS//-D_FORTIFY_SOURCE=[0-9]/}"
    export CFLAGS
  fi
}

function configure_legate()
{
  set -xou pipefail
  set +e

  # ${CC} is set by the conda compiler package. Disable shellcheck.
  # shellcheck disable=SC2154
  ./configure \
    --LEGATE_ARCH="${LEGATE_ARCH}" \
    --with-python \
    --with-cc="${CC}" \
    --with-cxx="${CXX}" \
    --build-march="${BUILD_MARCH}" \
    --cmake-generator="Ninja" \
    "${configure_args[@]}"

  ret=$?
  set -e
  if [[ "${ret}" != '0' ]]; then
    cat configure.log
    return "${ret}"
  fi
  if [[ "${LEGATE_BUILD_MODE:-}" != '' ]]; then
    found="$(grep -c -e "--build-type=${LEGATE_BUILD_MODE}" configure.log || true)"
    if [[ "${found}" == '0' ]]; then
      echo "FAILED TO PROPERLY SET BUILD TYPE:"
      echo "- expected to find --build-type=${LEGATE_BUILD_MODE} in configure.log"
      return 1
    fi
  fi
  return 0
}

function pip_install_legate()
{
  set -xeo pipefail
  # CPU_COUNT and PIP_CACHE_DIR are set by the build. Disable shellcheck.
  # shellcheck disable=SC2154
  export CMAKE_BUILD_PARALLEL_LEVEL="${CPU_COUNT}"
  # shellcheck disable=SC2154
  local cache_dir="${PIP_CACHE_DIR}"
  "${PYTHON}" -m pip install \
              --root / \
              --no-deps \
              --prefix "${PREFIX}" \
              --no-build-isolation \
              --cache-dir "${cache_dir}" \
              --disable-pip-version-check \
              . \
              -vv

  # Legion leaves an egg-info file which will confuse conda trying to pick up the information
  # Remove it so the legate is the only egg-info file added
  # SP_DIR is set by conda build. Disable shellcheck.
  # shellcheck disable=SC2154
  rm -rf "${SP_DIR}"/legion*egg-info

  # If building gex, for now remove legate MPI wrapper. This should be handled more completely
  # with a configure option in the future.
  if [[ ${LEGATE_NETWORK} == "gex" ]]; then
    find "${PREFIX}" -name "*legate*mpi*.so*" -exec rm {} \;
  fi
}

build_start=$(date)
echo "Build starting on ${build_start}"
run_command 'Preamble' preamble
run_command 'Configure Legate' configure_legate
run_command 'pip install Legate' pip_install_legate
build_end=$(date)
echo "Build ending on ${build_end}"
