#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

readonly DEFAULT_CONDUIT="ofi"
readonly DEFAULT_SYSTEM_CONFIG="slingshot11"
readonly DEFAULT_CUDA="ON"
# Threading suffix used by GASNet archives (libgasnet-<conduit>-<thread>.a)
readonly DEFAULT_THREADING="par"
# Pinned GASNet commit (matches what we validated on Perlmutter)
readonly GASNET_GITREF_SHA="e0af36ac9d3d632824be1851bcb3bc23bf05e489"

# Determine script directory dynamically
readonly SCRIPT_DIR="${CONDA_PREFIX}/gex-wrapper"

# Initialize variables with default values
conduit="${DEFAULT_CONDUIT}"
system_config="${DEFAULT_SYSTEM_CONFIG}"
cuda="${DEFAULT_CUDA}"
threading="${DEFAULT_THREADING}"
extra_linker_flags=""

# Help function to display usage
gex_wrapper_help() {
   echo "Usage: build-gex-wrapper [-h | --help] [-c conduit | --conduit conduit] [-s system_config | --system_config system_config] [-u ON/OFF | --use-cuda ON/OFF] [-f flags | --linker-flags \"<flags>\"]"
   echo "Build the Realm GASNet-EX wrapper in your conda environment."
   echo
   echo "Options:"
   echo "  -h, --help               Display this help and exit"
   echo "  -c, --conduit CONDUIT     GASNet conduit to use (default '${DEFAULT_CONDUIT}')"
   echo "  -s, --system_config SYS   System-specific configuration (default '${DEFAULT_SYSTEM_CONFIG}')"
   echo "  -u, --use-cuda ON/OFF     Enable (ON) or disable (OFF) CUDA (default '${DEFAULT_CUDA}')"
   echo "  -f, --linker-flags STR    Extra linker flags to append (default '-lhugetlbfs' when conduit='ofi' and system='slingshot11')"
   echo
}

# Parse command-line options (supporting both single-dash and double-dash)
ARGS=$(getopt -o hc:s:u:f: -l help,conduit:,system_config:,use-cuda:,linker-flags: -- "$@") || {
  gex_wrapper_help
  exit 1
}
eval set -- "${ARGS}"

while true; do
  case "$1" in
    -h | --help)
      gex_wrapper_help
      exit 0
      ;;
    -c | --conduit)
      conduit="$2"
      shift 2
      ;;
    -s | --system_config)
      system_config="$2"
      shift 2
      ;;
    -u | --use-cuda)
      cuda="$2"
      if [[ "${cuda}" != "ON" && "${cuda}" != "OFF" ]]; then
        echo "Invalid value for --use-cuda: must be ON or OFF" >&2
        exit 1
      fi
      shift 2
      ;;
    -f | --linker-flags)
      extra_linker_flags="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unexpected option: $1" >&2
      gex_wrapper_help
      exit 1
      ;;
  esac
done

# Default linker flags for Perlmutter OFI/slingshot11, unless overridden
if [[ -z "${extra_linker_flags}" && "${conduit}" == "ofi" && "${system_config}" == "slingshot11" ]]; then
  extra_linker_flags="-lhugetlbfs"
fi

# Ensure CONDA_PREFIX is set
if [[ -z "${CONDA_PREFIX}" ]]; then
  echo "Error: Please activate a conda environment before running this script."
  echo "Run:"
  echo "  $ conda activate <your-env-name>"
  echo "Then re-run this script."
  exit 1
fi

# Ensure cmake is available
if ! command -v cmake &>/dev/null; then
  echo "Error: cmake is not installed or not in PATH."
  echo "Please install it via your package manager or conda:"
  echo "  $ conda install -c conda-forge cmake"
  exit 1
fi

echo "Building GASNet-EX wrapper:"
echo "  Installation directory: ${CONDA_PREFIX}/lib"
echo "  Conduit: ${conduit}"
echo "  System configuration: ${system_config}"
echo "  CUDA enabled: ${cuda}"

# Proceed with the build process
if [[ ! -d "${SCRIPT_DIR}" ]]; then
  echo "Error: gex-wrapper directory '${SCRIPT_DIR}' not found."
  exit 1
fi

cd "${SCRIPT_DIR}" || { echo "Error: Failed to navigate to ${SCRIPT_DIR}"; exit 1; }
mkdir -p src/build
cd src/build || { echo "Error: Failed to navigate to build directory"; exit 1; }

CMAKE_ARGS=(
  -DLEGION_SOURCE_DIR="${SCRIPT_DIR}"
  -DREALM_SOURCE_DIR="${SCRIPT_DIR}"
  -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}"
  -DGASNet_CONDUIT="${conduit}"
  -DGASNet_SYSTEM="${system_config}"
  -DGEX_WRAPPER_BUILD_SHARED=ON
  -DGASNet_GITREF="${GASNET_GITREF_SHA}"
)

if [[ "${cuda}" == "ON" ]]; then
  # CI installs the minimal CUDA runtime/driver dev packages, not cuda-nvcc or
  # cuda-toolkit. Those packages provide headers/libs under targets/*-linux but
  # no activation hook, so fill GASNet configure vars while preserving user
  # overrides.
  cuda_target=""
  if [[ -d "${CONDA_PREFIX}/targets" ]]; then
    for target in "${CONDA_PREFIX}"/targets/*-linux; do
      if [[ -f "${target}/include/cuda.h" ]]; then
        cuda_target="${target}"
        break
      fi
    done
  fi
  if [[ -n "${cuda_target}" ]]; then
    export CUDA_HOME="${CUDA_HOME:-${cuda_target}}"
    export CUDA_CFLAGS="${CUDA_CFLAGS:--I${cuda_target}/include}"
    if [[ -d "${cuda_target}/lib/stubs" ]]; then
      export CUDA_LDFLAGS="${CUDA_LDFLAGS:--L${cuda_target}/lib/stubs}"
    else
      export CUDA_LDFLAGS="${CUDA_LDFLAGS:--L${cuda_target}/lib}"
    fi
    export CUDA_LIBS="${CUDA_LIBS:--lcuda}"
  fi

  CMAKE_ARGS+=(-DGASNet_CONFIGURE_ARGS="--enable-kind-cuda-uva")
fi

# Whole-archive embed of the conduit archive into the wrapper DSO.
# Note: libgasnet-<conduit>-par.a already contains gasnet_tools, so do NOT also
# link libgasnet_tools-par.a to avoid duplicate symbols.
GASNET_LIBDIR_EMBED="${SCRIPT_DIR}/src/build/embed-gasnet/install/lib"
MAIN_A="${GASNET_LIBDIR_EMBED}/libgasnet-${conduit}-${threading}.a"
LINK_FLAGS=("-Wl,--whole-archive,${MAIN_A},-no-whole-archive")

if [[ -n "${extra_linker_flags}" ]]; then
  read -r -a extra_linker_flags_array <<< "${extra_linker_flags}"
  LINK_FLAGS+=("${extra_linker_flags_array[@]}")
fi

CMAKE_ARGS+=(-DCMAKE_SHARED_LINKER_FLAGS="${LINK_FLAGS[*]}")

if ! cmake "${CMAKE_ARGS[@]}" ..; then
  gasnet_build_log="${SCRIPT_DIR}/src/build/embed-gasnet/build.log"
  if [[ -f "${gasnet_build_log}" ]]; then
    echo "GASNet build log:" >&2
    cat "${gasnet_build_log}" >&2
  fi
  exit 1
fi
cmake --build .
cmake --install .

echo
echo "Reactivate the conda environment to set necessary environment variables:"
echo
# shellcheck disable=SC2154
echo "  $ conda deactivate"
# shellcheck disable=SC2154
echo "  $ conda activate ${CONDA_DEFAULT_ENV}"
