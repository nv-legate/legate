#!/usr/bin/env bash

echo -e "\n\n--------------------- CONDA/CONDA-BUILD/BUILD.SH -----------------------\n"

set -xeo pipefail

# LICENSE, README.md, conda/, and configure are guaranteed to always be at the root
# directory. If we can't find them, then probably we are not in the root directory.
if [ ! -f LICENSE ] || [ ! -f README.md ] || [ ! -d conda ] || [ ! -f configure ]; then
  echo "Must run this script from the root directory"
  exit 1
fi

. continuous_integration/scripts/pretty_printing.bash

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
  CMAKE_ARGS="$(echo "${CMAKE_ARGS}" | ${SED} -r "s@_INCLUDE=ONLY@_INCLUDE=BOTH@g")"

  configure_args=()
  if [[ "${USE_OPENMP:-OFF}" == 'OFF' ]]; then
    configure_args+=(--with-openmp=0)
  else
    configure_args+=(--with-openmp)
  fi

  if [[ "${upload_enabled:-false}" == 'false' ]]; then
    configure_args+=(--with-tests)
    configure_args+=(--with-benchmarks)
  fi

  # We rely on an environment variable to determine if we need to build cpu-only bits
  if [[ "${CPU_ONLY:-0}" == '0' ]]; then
    configure_args+=(--with-cuda)
  else
    configure_args+=(--with-cuda=0)
  fi

  configure_args+=(--build-type="${LEGATE_BUILD_MODE}")

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

  export CUDAHOSTCXX="${CXX}"
  export OPENSSL_DIR="${PREFIX}"
  export CUDAFLAGS="-isystem ${PREFIX}/include -L${PREFIX}/lib"
  export LEGATE_DIR="$(${PYTHON} ./scripts/get_legate_dir.py)"
  export LEGATE_DIR
  export LEGATE_ARCH='arch-conda'
}

function configure_legate()
{
  set -xou pipefail
  set +e

  ./configure \
    --LEGATE_ARCH="${LEGATE_ARCH}" \
    --CUDAFLAGS="${CUDAFLAGS}" \
    --with-python \
    --with-cc="${CC}" \
    --with-cxx="${CXX}" \
    --build-march="${BUILD_MARCH}" \
    "${configure_args[@]}"

  ret=$?
  set -e
  if [[ "${ret}" != '0' ]]; then
    cat configure.log
    return ${ret}
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
  SKBUILD_BUILD_OPTIONS="-j${CPU_COUNT} VERBOSE=1" "${PYTHON}" -m pip install \
                       --root / \
                       --no-deps \
                       --prefix "${PREFIX}" \
                       --no-build-isolation \
                       --cache-dir "${PIP_CACHE_DIR}" \
                       --disable-pip-version-check \
                       . \
                       -vv

  # Legion leaves an egg-info file which will confuse conda trying to pick up the information
  # Remove it so the legate is the only egg-info file added
  rm -rf "${SP_DIR}"/legion*egg-info
}

echo "Build starting on $(date)"
run_command 'Preamble' preamble
run_command 'Configure Legate' configure_legate
run_command 'pip install Legate' pip_install_legate
echo "Build ending on $(date)"
