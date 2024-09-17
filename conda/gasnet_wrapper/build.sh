#!/usr/bin/env bash
echo -e "\n\n--------------------- CONDA/GASNET_WRAPPER/BUILD.SH -----------------------\n"

set -eo pipefail

# shellcheck disable=SC2154
mkdir "${PREFIX}/gex-wrapper"
# shellcheck disable=SC2154
mkdir "${PREFIX}/gex-wrapper/src"
# shellcheck disable=SC2154
cp -rv "${SRC_DIR}/cmake" "${PREFIX}/gex-wrapper"
# shellcheck disable=SC2154
cp -rv "${RECIPE_DIR}/build-gex-wrapper.sh" "${PREFIX}/gex-wrapper"
cp -rv "${SRC_DIR}"/runtime/realm/gasnetex/gasnetex_wrapper/* "${PREFIX}/gex-wrapper/src"

# Copy the [de]activate scripts to ${PREFIX}/etc/conda/[de]activate.d.
# This will allow them to be run on environment activation.
for CHANGE in "activate" "deactivate"
do
    mkdir -p "${PREFIX}/etc/conda/${CHANGE}.d"
    # shellcheck disable=SC2154
    cp "${RECIPE_DIR}/${CHANGE}.sh" "${PREFIX}/etc/conda/${CHANGE}.d/${PKG_NAME}_${CHANGE}.sh"
done
