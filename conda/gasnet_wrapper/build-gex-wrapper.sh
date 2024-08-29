#!/usr/bin/env bash

export SCRIPT_DIR="${CONDA_PREFIX}/gex-wrapper"

# Initialize variables with default values
conduit="ofi"  # Default conduit
system_config="slingshot11"  # Default system configuration

# Help function to display usage
gex_wrapper_help() {
   echo "Usage: build-gex-wrapper [-h] [-c conduit] [-s system_config]"
   echo "Build the Realm GASNet-EX wrapper in your conda environment."
   echo
   echo "Options:"
   echo "  -h                Display this help and exit"
   echo "  -c CONDUIT        Specify the GASNet conduit to use (default '${conduit}')"
   echo "  -s SYSTEM_CONFIG  Specify the system or machine-specific configuration (default '${system_config}')"
   echo
}

# Parse command-line options
while getopts ":hc:s:" opt; do
  case ${opt} in
    h)
      gex_wrapper_help
      exit 0
      ;;
    c)
      conduit="${OPTARG}"
      ;;
    s)
      system_config="${OPTARG}"
      ;;
    \?)
      echo "Invalid option: -${OPTARG}" >&2
      gex_wrapper_help
      exit 1
      ;;
    :)
      echo "Option -${OPTARG} requires an argument." >&2
      gex_wrapper_help
      exit 1
      ;;
  esac
done

# Check if CONDA_PREFIX is set
if [ -z "${CONDA_PREFIX}" ]; then
  echo "Please activate the environment in which to build the wrapper:"
  echo "\$ conda activate <your-env-name-here>"
  echo ""
  echo "Then re-run this script:"
  echo "\$  ${SHELL} ${SCRIPT_DIR}/conda/gasnet_wrapper/build-gex-wrapper.sh"
  exit 1
fi

echo "Building GASNet-EX wrapper:"
echo "  Installation directory: ${CONDA_PREFIX}/lib"
echo "  Conduit: ${conduit}"
echo "  System configuration: ${system_config}"

# Proceed with the build process
cd "${SCRIPT_DIR}" || { echo "Failed to navigate to gex-wrapper directory"; exit 1; }
mkdir -p src/build
cd src/build || { echo "Failed to navigate to build directory"; exit 1; }
cmake -DLEGION_SOURCE_DIR="${SCRIPT_DIR}" -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}" -DGASNet_CONDUIT="${conduit}" -DGASNet_SYSTEM="${system_config}" ..
cmake --build .
cmake --install .
cd ..
rm -rf build

echo
echo "Reactivate the conda environment to set the necessary environment variables:"
echo ""
echo "\$ conda deactivate"
echo "\$ conda activate ${CONDA_DEFAULT_ENV}"
