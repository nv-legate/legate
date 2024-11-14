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

list(APPEND CMAKE_MODULE_PATH "${LEGATE_CMAKE_DIR}/thirdparty/sphinx")

list(APPEND CMAKE_MESSAGE_CONTEXT "docs")

find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

file(GLOB_RECURSE legate_headers "${legate_LOCAL_INCLUDE_DIR}/*.h*")

# Only set "dynamic" settings here. Any settings which would have a hard-coded value
# should just be directly set in the Doxyfile.in
set(DOXYGEN_INPUT_DIR "${legate_LOCAL_INCLUDE_DIR}")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
set(DOXYGEN_INDEX_FILE "${DOXYGEN_OUTPUT_DIR}/xml/index.html")
set(DOXYGEN_STRIP_FROM_INC_PATH "${LEGATE_DIR}/src/cpp")
set(DOXYGEN_EXAMPLE_PATH "\"${LEGATE_DIR}/tests/cpp\" \"${LEGATE_DIR}/src/cpp\"")

set(DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

configure_file("${DOXYFILE_IN}" "${DOXYFILE_OUT}" @ONLY)

file(MAKE_DIRECTORY "${DOXYGEN_OUTPUT_DIR}") # Doxygen won't create this for us

add_custom_command(OUTPUT "${DOXYGEN_INDEX_FILE}"
                   DEPENDS "${legate_headers}"
                   COMMAND "${DOXYGEN_EXECUTABLE}" Doxyfile
                   MAIN_DEPENDENCY "${DOXYFILE_OUT}"
                   COMMENT "Generating doxygen output")

add_custom_target(Doxygen DEPENDS "${DOXYGEN_INDEX_FILE}" COMMENT "Running doxygen")

set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(SPHINX_BUILD "${CMAKE_CURRENT_BINARY_DIR}/sphinx")

file(COPY "${CMAKE_CURRENT_LIST_DIR}/eula.pdf" DESTINATION "${SPHINX_BUILD}/legate")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/eula.pdf"
     DESTINATION "${SPHINX_BUILD}/legate/latest")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/switcher.json"
     DESTINATION "${SPHINX_BUILD}/legate")

add_custom_target(Sphinx
                  COMMAND "${SPHINX_EXECUTABLE}" -b html
                          -Dbreathe_projects.legate="${DOXYGEN_OUTPUT_DIR}/xml" -W
                          "${SPHINX_SOURCE}" "${SPHINX_BUILD}/legate/latest"
                  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                  COMMENT "Generating documentation with Sphinx")
