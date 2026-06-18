#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
# Account for the move of the realm source.
if [[ -d "${SRC_DIR}"/runtime/realm/gasnetex/gasnetex_wrapper/ ]]; then
  cp -rv "${SRC_DIR}"/runtime/realm/gasnetex/gasnetex_wrapper/* "${PREFIX}/gex-wrapper/src"
else
  cp -rv "${SRC_DIR}/realm/src/realm/gasnetex/gasnetex_wrapper"/* "${PREFIX}/gex-wrapper/src"
fi

cmakelists="${PREFIX}/gex-wrapper/src/CMakeLists.txt"
if grep -q "find_package(GASNet REQUIRED)" "${cmakelists}" && \
   ! grep -q "include(FetchAndBuildGASNet)" "${cmakelists}"; then
  tmp_cmakelists="$(mktemp)"
  awk '
    /find_package\(GASNet REQUIRED\)/ {
      print "find_package(GASNet)"
      print "if(NOT GASNet_FOUND)"
      print "  set(GASNet_BUILD_SHARED TRUE)"
      print "  include(FetchAndBuildGASNet)"
      print "  find_package(GASNet REQUIRED)"
      print "endif()"
      next
    }
    { print }
  ' "${cmakelists}" > "${tmp_cmakelists}"
  mv "${tmp_cmakelists}" "${cmakelists}"
fi

# Copy the [de]activate scripts to ${PREFIX}/etc/conda/[de]activate.d.
# This will allow them to be run on environment activation.
for CHANGE in "activate" "deactivate"
do
    mkdir -p "${PREFIX}/etc/conda/${CHANGE}.d"
    # shellcheck disable=SC2154
    cp "${RECIPE_DIR}/${CHANGE}.sh" "${PREFIX}/etc/conda/${CHANGE}.d/${PKG_NAME}_${CHANGE}.sh"
done
