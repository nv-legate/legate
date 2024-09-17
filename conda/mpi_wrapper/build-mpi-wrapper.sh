#!/usr/bin/env bash

export SCRIPT_DIR="${CONDA_PREFIX}/mpi-wrapper"

# Initialize variables with default values
compiler="CC"  # Default compiler

# Help function to display usage
mpi_wrapper_help() {
   echo "Usage: build-mpi-wrapper [-h] [-c COMPILER]"
   echo "Build the Legate MPI wrapper in your conda environment."
   echo ""
   echo "Options:"
   echo "  -h                Display this help and exit"
   echo "  -c COMPILER       Specify the compiler to use (default '${compiler}')"
   echo ""
}

# Parse command-line options
while getopts ":hc:s:" opt; do
  case ${opt} in
    h)
      mpi_wrapper_help
      exit 0
      ;;
    c)
      compiler="${OPTARG}"
      ;;
    \?)
      echo "Invalid option: -${OPTARG}" >&2
      mpi_wrapper_help
      exit 1
      ;;
    :)
      echo "Option -${OPTARG} requires an argument." >&2
      mpi_wrapper_help
      exit 1
      ;;
    *)
      echo "Invalid option: -${OPTARG}" >&2
      exit 1
      ;;
  esac
done

# Check if CONDA_PREFIX is set
if [[ -z "${CONDA_PREFIX}" ]]; then
  echo "Please activate the environment in which to build the wrapper:"
  echo "\$ conda activate <your-env-name-here>"
  echo ""
  echo "Then re-run this script:"
  echo "\$ ${SHELL} ${SCRIPT_DIR}/conda/mpi-wrapper/build-mpi-wrapper.sh"
  exit 1
fi

echo "Building Legate MPI wrapper:"
echo "  Installation directory: ${CONDA_PREFIX}/lib"
echo "  Compiler: ${compiler}"

# Proceed with the build process
cd "${SCRIPT_DIR}" || { echo "Failed to navigate to mpi-wrapper directory"; exit 1; }
CXX="${compiler}" PREFIX="${SCRIPT_DIR}" ./install.bash

echo ""
echo "Reactivate the conda environment to set the necessary environment variables:"
echo ""
echo "\$ conda deactivate"
# shellcheck disable=SC2154
echo "\$ conda activate ${CONDA_DEFAULT_ENV}"
