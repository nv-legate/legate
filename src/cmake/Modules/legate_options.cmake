#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "options")

function(_legate_option_or_setting kind name var_env docs default)
  macro(_set_val value)
    if(kind STREQUAL "OPTION")
      option(${name} "${docs}" "${value}")
    elseif(kind STREQUAL "SETTING")
      set(${name} ${value} CACHE STRING "${docs}")
    else()
      message(FATAL_ERROR "Unknown kind: ${kind}")
    endif()
    set(${name} ${value})
  endmacro()

  if(default STREQUAL "UNSET")
    unset(default)
  elseif(default STREQUAL "SET_BUT_EMPTY")
    set(default "")
  endif()

  if(DEFINED ${name})
    _set_val("${${name}}")
    set(providence "predefined")
  elseif(DEFINED ENV{${var_env}})
    _set_val("$ENV{${var_env}}")
    set(providence "environment '${var_env}'")
  elseif(DEFINED default)
    _set_val("${default}")
    set(providence "default value")
  else()
    message(VERBOSE "Not setting ${name}")
    return()
  endif()

  if(NOT DEFINED ${name})
    message(FATAL_ERROR "Bug in this function")
  endif()

  message(VERBOSE "${name}=${${name}} (from ${providence})")
  set(${name} ${${name}} PARENT_SCOPE)
endfunction()

function(legate_setting name var_env docs default_val)
  _legate_option_or_setting(SETTING "${name}" "${var_env}" "${docs}" "${default_val}")
  if(DEFINED ${name})
    set(${name} ${${name}} PARENT_SCOPE)
  endif()
endfunction()

function(legate_option name var_env docs default_val)
  _legate_option_or_setting(OPTION "${name}" "${var_env}" "${docs}" "${default_val}")
  if(DEFINED ${name})
    set(${name} ${${name}} PARENT_SCOPE)
  endif()
endfunction()

# Initialize these vars from the CLI, then fallback to an evar or a default value.
legate_option(legate_BUILD_TESTS BUILD_TESTS "Whether to build the C++ tests" OFF)
legate_option(legate_BUILD_EXAMPLES BUILD_EXAMPLES
              "Whether to build the C++/python examples" OFF)
legate_option(legate_BUILD_DOCS BUILD_DOCS "Build doxygen docs" OFF)
legate_option(Legion_SPY USE_SPY "Enable detailed logging for Legion Spy" OFF)
legate_option(Legion_USE_LLVM USE_LLVM "Use LLVM JIT operations" OFF)
legate_option(Legion_USE_CUDA USE_CUDA "Enable Legion support for the CUDA runtime" OFF)
legate_option(Legion_USE_HDF5 USE_HDF "Enable support for HDF5" OFF)
legate_setting(Legion_NETWORKS NETWORKS
               "Networking backends to use (semicolon-separated)" SET_BUT_EMPTY)
legate_option(Legion_USE_OpenMP USE_OPENMP "Use OpenMP" OFF)
legate_option(Legion_USE_Python LEGION_USE_PYTHON "Use Python" OFF)
legate_option(Legion_BOUNDS_CHECKS CHECK_BOUNDS
              "Enable bounds checking in Legion accessors" OFF)
legate_option(legate_SKIP_NVCC_PEDANTIC_CHECK LEGATE_SKIP_NVCC_PEDANTIC_CHECK
              "Skip checking for -pedantic or -Wpedantic compiler flags for NVCC" OFF)
legate_option(legate_ENABLE_SANITIZERS LEGATE_ENABLE_SANITIZERS
              "Enable sanitizer support for legate" OFF)
legate_option(legate_IGNORE_INSTALLED_PACKAGES
              LEGATE_IGNORE_INSTALLED_PACKAGES
              "When deciding to search for or download third-party packages, never search and always download"
              OFF)
legate_option(legate_USE_CAL LEGATE_USE_CAL "Enable CAL support in Legate" OFF)
legate_option(legate_BUILD_BENCHMARKS LEGATE_BUILD_BENCHMARKS "Build legate benchmarks"
              OFF)
legate_option(legate_USE_CPROFILE LEGATE_USE_CPROFILE "Enable Cprofile in Legate" OFF)

if("${Legion_NETWORKS}" MATCHES ".*gasnet(1|ex).*")
  legate_setting(GASNet_ROOT_DIR GASNET "GASNet root directory" UNSET)
  legate_setting(GASNet_CONDUIT CONDUIT "Default GASNet conduit" "mpi")

  if(NOT GASNet_ROOT_DIR)
    legate_option(Legion_EMBED_GASNet LEGION_EMBED_GASNET
                  "Embed a custom GASNet build into Legion" ON)
  endif()
endif()

legate_setting(Legion_MAX_DIM LEGION_MAX_DIM "Maximum dimension" 4)

# Check the max dimensions
if((Legion_MAX_DIM LESS 1) OR (Legion_MAX_DIM GREATER 9))
  message(FATAL_ERROR "The maximum number of Legate dimensions must be between"
                      " 1 and 9 inclusive")
endif()

legate_setting(Legion_MAX_FIELDS LEGION_MAX_FIELDS "Maximum number of fields" 256)

# Check that max fields is between 32 and 4096 and is a power of 2
if(NOT Legion_MAX_FIELDS MATCHES "^(32|64|128|256|512|1024|2048|4096)$")
  message(FATAL_ERROR "The maximum number of Legate fields must be a power of 2"
                      " between 32 and 4096 inclusive")
endif()

legate_setting(CMAKE_CUDA_RUNTIME_LIBRARY CMAKE_CUDA_RUNTIME_LIBRARY
               "Default linkage kind for CUDA" SHARED)
legate_setting(NCCL_DIR NCCL_DIR "NCCL Root directory" UNSET)
legate_setting(CUDA_TOOLKIT_ROOT_DIR CUDA "CUDA Root directory" UNSET)

legate_setting(legate_CXX_FLAGS LEGATE_CXX_FLAGS "C++ flags for legate" SET_BUT_EMPTY)
legate_setting(legate_CUDA_FLAGS LEGATE_CUDA_FLAGS "CUDA flags for legate" SET_BUT_EMPTY)
legate_setting(legate_LINKER_FLAGS LEGATE_LD_FLAGS "Linker flags for legate"
               SET_BUT_EMPTY)

legate_setting(Legion_CXX_FLAGS LEGION_CXX_FLAGS "C++ flags for Legion" SET_BUT_EMPTY)
legate_setting(Legion_CUDA_FLAGS LEGION_CUDA_FLAGS "CUDA flags for Legion" SET_BUT_EMPTY)
legate_setting(Legion_LINKER_FLAGS LEGION_LD_FLAGS "Linker flags for Legion"
               SET_BUT_EMPTY)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
