#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(legate_generate_install_info_py)
  list(APPEND CMAKE_MESSAGE_CONTEXT "generate_install_info_py")
  # Legion sets this to "OFF" if not enabled, normalize it to an empty list instead
  if(NOT Legion_NETWORKS)
    set(Legion_NETWORKS "")
  endif()

  # Set by the pip wheels if they are being built, set to OFF if not
  if(NOT LEGATE_BUILD_PIP_WHEELS)
    set(LEGATE_BUILD_PIP_WHEELS OFF)
  endif()

  # CMake renders spaces in substitutions as "\ " which is a python syntax error. However,
  # converting the value to a CMake list (using semicolons) renders in substitutions as a
  # space-delimited string, which is what is desired. It is unclear if this CMake behavior
  # is documented/intendeed, but it it unlikely to change. Other approaches, e.g. VERBATIM
  # for the custom target, fail for other reasons.
  string(REPLACE " " ";" legate_configure_options_list "${LEGATE_CONFIGURE_OPTIONS}")

  add_custom_target(generate_install_info_py ALL
                    COMMAND ${CMAKE_COMMAND} -DLEGATE_CMAKE_DIR="${LEGATE_CMAKE_DIR}"
                            -DLEGATE_ARCH="${LEGATE_ARCH}" -DLEGATE_DIR="${LEGATE_DIR}"
                            -DLEGATE_BUILD_PIP_WHEELS="${LEGATE_BUILD_PIP_WHEELS}"
                            -DLegion_VERSION="${Legion_VERSION}"
                            -DLegion_GIT_BRANCH="${Legion_GIT_BRANCH}"
                            -DLegion_GIT_REPO="${Legion_GIT_REPO}"
                            -DLegion_NETWORKS="${Legion_NETWORKS}"
                            -DGASNet_CONDUIT="${GASNet_CONDUIT}"
                            -DLegion_USE_CUDA="${Legion_USE_CUDA}"
                            -DLegion_USE_OpenMP="${Legion_USE_OpenMP}"
                            -DLegion_MAX_DIM="${Legion_MAX_DIM}"
                            -DLegion_MAX_FIELDS="${Legion_MAX_FIELDS}"
                            -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
                            -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
                            -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
                            -DLEGATE_CONFIGURE_OPTIONS="${legate_configure_options_list}"
                            -Dlegate_LIB_NAME="$<TARGET_FILE_PREFIX:legate::legate>$<TARGET_FILE_BASE_NAME:legate::legate>"
                            -Dlegate_FULL_LIB_NAME="$<TARGET_FILE_NAME:legate::legate>" -P
                            "${LEGATE_CMAKE_DIR}/generate_install_info_py.cmake"
                    DEPENDS "${LEGATE_CMAKE_DIR}/templates/install_info.py.in"
                    BYPRODUCTS "${LEGATE_DIR}/src/python/legate/install_info.py"
                    COMMENT "Generate install_info.py")

endfunction()
