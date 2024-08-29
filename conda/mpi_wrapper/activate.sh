#!/usr/bin/env bash
echo -e "\n\n--------------------- CONDA/MPI_WRAPPER/ACTIVATE.SH -----------------------\n"

set -eo pipefail
wrapper_file=$(find "${CONDA_PREFIX}/mpi-wrapper" -regex ".*/liblgcore_mpi_wrapper\.\(so\|dylib\)" -print -quit)
export LEGATE_MPI_WRAPPER="${wrapper_file}"
echo "LEGATE_MPI_WRAPPER=${LEGATE_MPI_WRAPPER}"
