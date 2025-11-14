#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(validate_nccl_version)
  if(NOT legate_nccl_include_dir)
    get_target_property(nccl_include_dirs NCCL::NCCL INTERFACE_INCLUDE_DIRECTORIES)

    set(candidate_incdirs)
    foreach(potential_dir IN LISTS nccl_include_dirs)
      string(GENEX_STRIP "${potential_dir}" stripped_dir)
      list(APPEND candidate_incdirs "${stripped_dir}")
    endforeach()

    find_path(legate_nccl_include_dir
              NAMES nccl.h
              PATH_SUFFIXES nccl
              HINTS ${candidate_incdirs}
              DOC "Path containing nccl.h")
  endif()

  if(NOT legate_nccl_include_dir)
    message(FATAL_ERROR "Could not find nccl.h in any include directory for "
                        "NCCL version validation. Searched: ${candidate_incdirs}")
  endif()

  set(nccl_header_path "${legate_nccl_include_dir}/nccl.h")

  file(STRINGS "${nccl_header_path}" maj_line LIMIT_COUNT 1
       REGEX [=[^#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+]=])
  file(STRINGS "${nccl_header_path}" min_line LIMIT_COUNT 1
       REGEX [=[^#define[ \t]+NCCL_MINOR[ \t]+[0-9]+]=])

  string(REGEX MATCH [=[[0-9]+]=] nccl_major "${maj_line}")
  string(REGEX MATCH [=[[0-9]+]=] nccl_minor "${min_line}")

  if(nccl_major STREQUAL "" OR nccl_minor STREQUAL "")
    message(FATAL_ERROR "Could not read NCCL version from ${nccl_header_path}")
  endif()

  set(nccl_version "${nccl_major}.${nccl_minor}")
  # If you bump this, also bump it on all dependency lists.
  set(required_nccl_version 2.28)

  if("${required_nccl_version}" VERSION_LESS "${nccl_version}")
    message(FATAL_ERROR "Detected NCCL version ${nccl_version}, but "
                        "version ${required_nccl_version} or lower is required.")
  endif()

  message(STATUS "NCCL version ${nccl_version} meets requirement <= ${required_nccl_version}"
  )
endfunction()

function(find_or_configure_nccl)
  list(APPEND CMAKE_MESSAGE_CONTEXT "nccl")

  if(TARGET NCCL::NCCL)
    validate_nccl_version()
    return()
  endif()

  rapids_find_generate_module(NCCL HEADER_NAMES nccl.h LIBRARY_NAMES nccl)
  # Currently NCCL has no CMake build-system so we require it built and installed on the
  # machine already
  rapids_find_package(NCCL)
  if(TARGET NCCL::NCCL)
    validate_nccl_version()
    return()
  endif()

  # If the user has installed NCCL to a system location, then CMake might not find their
  # NCCL because it will be under /usr/lib/<whatever>. CMake will, however, search under
  # there if CMAKE_LIBRARY_ARCHITECTURE is set, so we need to ensure that it is.
  if(CMAKE_LIBRARY_ARCHITECTURE)
    # In this case, the below find_package() will fail (we are doing the same thing as
    # above without changing anything), so better to fail with a more useful error message
    # instead.
    message(VERBOSE "CMAKE_LIBRARY_ARCHITECTURE is set: ${CMAKE_LIBRARY_ARCHITECTURE}")
    message(FATAL_ERROR "Could not find NCCL on your system. It's possible that you have it installed in a location that CMake does not know to search for automatically. Please report this case to Legate maintainers."
    )
  endif()

  message(VERBOSE "CMAKE_LIBRARY_ARCHITECTURE is unset, attempting to deduce it")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(VERBOSE "linux system detected")
    foreach(arch_dir IN ITEMS x86_64-linux-gnu aarch64-linux-gnu)
      if(EXISTS "/usr/lib/${arch_dir}")
        set(CMAKE_LIBRARY_ARCHITECTURE "${arch_dir}")
        break()
      endif()
    endforeach()
  endif()

  if(NOT CMAKE_LIBRARY_ARCHITECTURE)
    message(FATAL_ERROR "Could not auto-deduce CMAKE_LIBRARY_ARCHITECTURE while trying to locate NCCL. It's possible that you have NCCL installed in a location that CMake does not know to search for automatically (which we tried to remedy by deducing CMAKE_LIBRARY_ARCHITECTURE). Please report this case to Legate maintainers."
    )
  endif()

  message(VERBOSE "CMAKE_LIBRARY_ARCHITECTURE is set: ${CMAKE_LIBRARY_ARCHITECTURE}")
  rapids_find_package(NCCL REQUIRED)
  validate_nccl_version()
endfunction()
