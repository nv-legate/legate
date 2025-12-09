#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(
  APPEND CMAKE_MODULE_PATH
  "${LEGATE_CMAKE_DIR}/thirdparty/sphinx" # codespell:ignore thirdparty
)

list(APPEND CMAKE_MESSAGE_CONTEXT "docs")

find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

# Only set "dynamic" settings here. Any settings which would have a hard-coded value
# should just be directly set in the Doxyfile.in
set(DOXYGEN_INPUT_DIR "${legate_LOCAL_INCLUDE_DIR}")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
set(DOXYGEN_INDEX_FILE "${DOXYGEN_OUTPUT_DIR}/xml/index.html")
set(DOXYGEN_STRIP_FROM_INC_PATH "${LEGATE_DIR}/src/cpp")
set(DOXYGEN_EXAMPLE_PATH "${LEGATE_DIR}/tests/cpp,${LEGATE_DIR}/src/cpp")

set(DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

configure_file("${DOXYFILE_IN}" "${DOXYFILE_OUT}" @ONLY)

file(MAKE_DIRECTORY "${DOXYGEN_OUTPUT_DIR}") # Doxygen won't create this for us

add_custom_command(
  OUTPUT "${DOXYGEN_INDEX_FILE}"
  DEPENDS "${DOXYFILE_OUT}"
  COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYFILE_OUT}"
  COMMENT "Generating doxygen output"
)

add_custom_target(Doxygen DEPENDS "${DOXYGEN_INDEX_FILE}" COMMENT "Running doxygen")

set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(SPHINX_BUILD "${CMAKE_CURRENT_BINARY_DIR}/sphinx")

file(
  COPY "${CMAKE_CURRENT_SOURCE_DIR}/switcher.json"
  DESTINATION "${SPHINX_BUILD}/legate"
)

if(NOT legate_LICENSE_FILE)
  message(
    FATAL_ERROR
    "Don't know where Legate license file is. "
    "Expected cmake variable legate_LICENSE_FILE to contain it's location, but that variable is empty. "
    "Presumably somebody changed the name?"
  )
endif()

# Need to make this every time because git does not allow you to check in empty
# directories
file(MAKE_DIRECTORY "${SPHINX_SOURCE}/generated")

# Need to do this symlink because the .rst files have no way to know where the build
# directory is. I tried using rst_prolog (while setting an environment variable
# LEGATE_SPHINX_BUILD):
#
# rst_prolog="""
# .. |BUILD_DIR| replace:: {os.environ["LEGATE_SPHINX_BUILD"]}
# """
#
# and then:
#
# .. include:: |BUILD_DIR|/path/to/license.txt
#
# But this didn't work because you can't do replacements inside other directives in rst
# apparently. The only way I found that works is just symlinking it somewhere into the
# source directory then...
add_custom_target(
  symlink_license_file
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink "${legate_LICENSE_FILE}"
    "${SPHINX_SOURCE}/generated/licenses.txt"
  BYPRODUCTS "${SPHINX_SOURCE}/generated/licenses.txt"
  COMMENT "Symlinking licenses.txt -> ${SPHINX_SOURCE}/generated"
)
add_dependencies(symlink_license_file legate_write_licenses)

cmake_path(
  SET generated_cpp_api_dir
  NORMALIZE
  "${CMAKE_CURRENT_BINARY_DIR}/generated_cpp_api"
)
file(MAKE_DIRECTORY "${generated_cpp_api_dir}")
add_custom_target(
  generate_cpp_api_rst
  COMMAND
    ${LEGATE_DIR}/scripts/maint/gen_api_doc.py "${DOXYGEN_OUTPUT_DIR}/xml"
    --generate=class,struct,group --output-dir="${generated_cpp_api_dir}" --force
  WORKING_DIRECTORY "${generated_cpp_api_dir}"
  COMMENT "Generating C++ API rst files"
)
add_dependencies(generate_cpp_api_rst Doxygen)

add_custom_target(
  symlink_cpp_api_dir
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink "${generated_cpp_api_dir}"
    "${SPHINX_SOURCE}/api/cpp/generated"
  COMMENT "Symlinking ${generated_cpp_api_dir} -> ${SPHINX_SOURCE}/api/cpp/generated"
)
add_dependencies(symlink_cpp_api_dir generate_cpp_api_rst)

add_custom_target(
  Sphinx
  COMMAND
    "${SPHINX_EXECUTABLE}" -b html -Dbreathe_projects.legate="${DOXYGEN_OUTPUT_DIR}/xml"
    -W "${SPHINX_SOURCE}" "${SPHINX_BUILD}/legate/latest"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Generating documentation with Sphinx"
)
add_dependencies(Sphinx symlink_cpp_api_dir symlink_license_file)
