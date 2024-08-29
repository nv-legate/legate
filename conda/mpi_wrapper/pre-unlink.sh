#!/usr/bin/env bash
echo -e "\n\n--------------------- CONDA/MPI_WRAPPER/PREUNLINK.SH -----------------------\n"

set -eo pipefail
echo "Remove the built artifacts"
MPI_WRAPPER_DIR="${CONDA_PREFIX}/mpi-wrapper"
rm -rf "${MPI_WRAPPER_DIR}/include" "${MPI_WRAPPER_DIR}/lib*"
