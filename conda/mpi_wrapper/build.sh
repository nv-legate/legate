#!/usr/bin/env bash
echo -e "\n\n--------------------- CONDA/MPI_WRAPPER/BUILD.SH -----------------------\n"

set -eou pipefail

# shellcheck disable=SC2154
cp -rv "${SRC_DIR}/share/legate/mpi_wrapper" "${PREFIX}/mpi-wrapper"
# shellcheck disable=SC2154
cp -rv "${RECIPE_DIR}/build-mpi-wrapper.sh" "${PREFIX}/mpi-wrapper"

# Copy the [de]activate scripts to ${PREFIX}/etc/conda/[de]activate.d.
# This will allow them to be run on environment activation.
for CHANGE in "activate" "deactivate"
do
    mkdir -p "${PREFIX}/etc/conda/${CHANGE}.d"
    # shellcheck disable=SC2154
    cp "${RECIPE_DIR}/${CHANGE}.sh" "${PREFIX}/etc/conda/${CHANGE}.d/${PKG_NAME}_${CHANGE}.sh"
done
