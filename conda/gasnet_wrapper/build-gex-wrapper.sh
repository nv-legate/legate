#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

readonly DEFAULT_CONDUIT="ofi"
readonly DEFAULT_SYSTEM_CONFIG="slingshot11"
readonly DEFAULT_CUDA="ON"

# Determine script directory dynamically
readonly SCRIPT_DIR="${CONDA_PREFIX}/gex-wrapper"

# Initialize variables with default values
conduit="${DEFAULT_CONDUIT}"
system_config="${DEFAULT_SYSTEM_CONFIG}"
cuda="${DEFAULT_CUDA}"

# Help function to display usage
gex_wrapper_help() {
   echo "Usage: build-gex-wrapper [-h | --help] [-c conduit | --conduit conduit] [-s system_config | --system_config system_config] [-u ON/OFF | --use-cuda ON/OFF]"
   echo "Build the Realm GASNet-EX wrapper in your conda environment."
   echo
   echo "Options:"
   echo "  -h, --help               Display this help and exit"
   echo "  -c, --conduit CONDUIT     Specify the GASNet conduit to use (default '${DEFAULT_CONDUIT}')"
   echo "  -s, --system_config SYS   Specify the system-specific configuration (default '${DEFAULT_SYSTEM_CONFIG}')"
   echo "  -u, --use-cuda ON/OFF     Enable (ON) or disable (OFF) CUDA (default '${DEFAULT_CUDA}')"
   echo
}

# Parse command-line options (supporting both single-dash and double-dash)
ARGS=$(getopt -o hc:s:u: -l help,conduit:,system_config:,use-cuda: -- "$@") || {
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

# Ensure CONDA_PREFIX is set
if [[ -z "${CONDA_PREFIX}" ]]; then
  echo "Error: Please activate a conda environment before running this script."
  echo "Run:"
  echo "  \$ conda activate <your-env-name>"
  echo "Then re-run this script."
  exit 1
fi

# Ensure cmake is available
if ! command -v cmake &>/dev/null; then
  echo "Error: cmake is not installed or not in PATH."
  echo "Please install it via your package manager or conda:"
  echo "  \$ conda install -c conda-forge cmake"
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
  -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}"
  -DGASNet_CONDUIT="${conduit}"
  -DGASNet_SYSTEM="${system_config}"
  -DGEX_WRAPPER_BUILD_SHARED=ON
)

if [[ "${cuda}" == "ON" ]]; then
  CMAKE_ARGS+=(-DGASNet_CONFIGURE_ARGS="--enable-kind-cuda-uva")
fi

cmake "${CMAKE_ARGS[@]}" ..
cmake --build .
cmake --install .

echo
echo "Reactivate the conda environment to set necessary environment variables:"
echo
echo "  \$ conda deactivate"
# shellcheck disable=SC2154
echo "  \$ conda activate ${CONDA_DEFAULT_ENV}"
