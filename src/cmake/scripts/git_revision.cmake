#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

foreach(var GIT_EXECUTABLE TEMPLATE_FILE SRC_DIRS PREFIXES DEST_FILES CACHE_FILES)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "Must also pass ${var}")
  endif()
endforeach()

separate_arguments(SRC_DIRS) # cmake-lint: disable=E1120
separate_arguments(PREFIXES) # cmake-lint: disable=E1120
separate_arguments(DEST_FILES) # cmake-lint: disable=E1120
separate_arguments(CACHE_FILES) # cmake-lint: disable=E1120

list(LENGTH SRC_DIRS num_src)
list(LENGTH PREFIXES num_prefix)
list(LENGTH DEST_FILES num_dest)
list(LENGTH CACHE_FILES num_cache)
if(NOT num_src EQUAL num_prefix)
  message(FATAL_ERROR "Must pass same number of prefixes as src dirs: "
                      "(${num_src} src dirs, ${num_prefix} prefixes)")
endif()
if(NOT num_src EQUAL num_dest)
  message(FATAL_ERROR "Must pass same number of dest files as src dirs: "
                      "(${num_src} src dirs, ${num_dest} dest files)")
endif()
if(NOT num_src EQUAL num_cache)
  message(FATAL_ERROR "Must pass same number of cache files as src dirs: "
                      "(${num_src} src dirs, ${num_cache} cache files)")
endif()

foreach(source_dir PREFIX dest_file cache_file IN ZIP_LISTS SRC_DIRS PREFIXES DEST_FILES
                   CACHE_FILES)
  if(source_dir)
    execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
                    WORKING_DIRECTORY "${source_dir}" OUTPUT_VARIABLE GIT_HASH
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    set(GIT_HASH "<unknown git hash>")
  endif()

  if(EXISTS ${cache_file})
    file(READ "${cache_file}" cached_value)
    if("${cached_value}" STREQUAL "${GIT_HASH}")
      continue()
    endif()
  else()
    file(WRITE "${cache_file}" "${GIT_HASH}")
  endif()

  configure_file("${TEMPLATE_FILE}" "${dest_file}" @ONLY)
endforeach()
