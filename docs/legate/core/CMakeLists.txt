#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

list(APPEND CMAKE_MODULE_PATH "${LEGATE_CORE_DIR}/cmake/thirdparty/sphinx")

list(APPEND CMAKE_MESSAGE_CONTEXT "docs")

find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

file(GLOB_RECURSE legate_core_headers "${legate_core_LOCAL_INCLUDE_DIR}/*.h*")

set(DOXYGEN_INPUT_DIR "${legate_core_LOCAL_INCLUDE_DIR}")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
set(DOXYGEN_INDEX_FILE "${DOXYGEN_OUTPUT_DIR}/xml/index.html")
set(DOXYGEN_PROJECT_NAME "Legate")
set(DOXYGEN_FULL_PATH_NAMES NO)
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_EXTENSION_MAPPING "cu=C++ cuh=C++")
set(DOXYGEN_HIDE_UNDOC_MEMBERS YES)
set(DOXYGEN_HIDE_UNDOC_CLASSES YES)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_MATHJAX_VERSION MathJax_3)
set(DOXYGEN_STRIP_FROM_INC_PATH "${LEGATE_CORE_DIR}/src")
set(DOXYGEN_EXAMPLE_PATH "\"${LEGATE_CORE_DIR}/tests/cpp\" \"${LEGATE_CORE_DIR}/src\"")
set(DOXYGEN_QUIET YES)
set(DOXYGEN_PREDEFINED
    [=[
  "DOXYGEN" \
  "LEGATE_DOXYGEN=1" \
  "LEGATE_HOST_DEVICE" \
  "LEGATE_USE_CUDA=1" \
  "LEGATE_USE_OPENMP=1" \
  "LEGATE_STL_REQUIRES=requires" \
  "LEGATE_DEFINED(X)=defined(X) && X" \
  "LEGATE_STL_UNSPECIFIED(...)=unspecified"
  ]=])
set(DOXYGEN_EXPAND_AS_DEFINED [=[
  ]=])
set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_INLINE_INFO NO)

set(DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

configure_file("${DOXYFILE_IN}" "${DOXYFILE_OUT}" @ONLY)

file(MAKE_DIRECTORY "${DOXYGEN_OUTPUT_DIR}") # Doxygen won't create this for us

add_custom_command(OUTPUT "${DOXYGEN_INDEX_FILE}"
                   DEPENDS "${legate_core_headers}"
                   COMMAND "${DOXYGEN_EXECUTABLE}" Doxyfile
                   MAIN_DEPENDENCY "${DOXYFILE_OUT}"
                   COMMENT "Generating doxygen output")

add_custom_target(Doxygen DEPENDS "${DOXYGEN_INDEX_FILE}" COMMENT "Running doxygen")

set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(SPHINX_BUILD "${CMAKE_CURRENT_BINARY_DIR}/sphinx")

file(COPY "${LEGATE_CORE_DIR}/eula.pdf" DESTINATION "${SPHINX_BUILD}/legate")
file(COPY "${LEGATE_CORE_DIR}/eula.pdf" DESTINATION "${SPHINX_BUILD}/legate/latest")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/switcher.json"
     DESTINATION "${SPHINX_BUILD}/legate")

add_custom_target(Sphinx
                  COMMAND "${SPHINX_EXECUTABLE}" -b html
                          -Dbreathe_projects.legate_core="${DOXYGEN_OUTPUT_DIR}/xml" -W
                          "${SPHINX_SOURCE}" "${SPHINX_BUILD}/legate/latest"
                  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                  COMMENT "Generating documentation with Sphinx")
