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

include_guard(GLOBAL)

# cmake/Modules/cuda_arch_helpers.cmake:16: [R0912] Too many branches 14/12
#
# Many branches are OK for this function, not only is it clear what they are doing, but
# doing this in a branchless way would be less readable.
#
# cmake-lint: disable=R0912,R0915
function(legate_convert_cuda_arch_from_names)
  list(APPEND CMAKE_MESSAGE_CONTEXT "set_cuda_arch_from_names")

  set(options)
  set(oneValueArgs INPUT_VAR DEST_VAR)
  set(multiValueArgs)
  cmake_parse_arguments(_LEGATE "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  if(NOT _LEGATE_INPUT_VAR)
    message(FATAL_ERROR "Must pass INPUT_VAR")
  endif()

  if(NOT _LEGATE_DEST_VAR)
    set(${_LEGATE_DEST_VAR} ${_LEGATE_INPUT_VAR})
  endif()

  set(cuda_archs "")
  # translate legacy arch names into numbers
  if(${_LEGATE_INPUT_VAR} MATCHES "fermi")
    list(APPEND cuda_archs 20)
  endif()
  if(${_LEGATE_INPUT_VAR} MATCHES "kepler")
    list(APPEND cuda_archs 30)
  endif()
  if(${_LEGATE_INPUT_VAR} MATCHES "k20")
    list(APPEND cuda_archs 35)
  endif()
  if(${_LEGATE_INPUT_VAR} MATCHES "k80")
    list(APPEND cuda_archs 37)
  endif()
  if(${_LEGATE_INPUT_VAR} MATCHES "maxwell")
    list(APPEND cuda_archs 52)
  endif()
  if(${_LEGATE_INPUT_VAR} MATCHES "pascal")
    list(APPEND cuda_archs 60)
  endif()
  if(${_LEGATE_INPUT_VAR} MATCHES "volta")
    list(APPEND cuda_archs 70)
  endif()
  if(${_LEGATE_INPUT_VAR} MATCHES "turing")
    list(APPEND cuda_archs 75)
  endif()
  if(${_LEGATE_INPUT_VAR} MATCHES "ampere")
    list(APPEND cuda_archs 80)
  endif()
  if(${_LEGATE_INPUT_VAR} MATCHES "ada")
    list(APPEND cuda_archs 89)
  endif()
  if(${_LEGATE_INPUT_VAR} MATCHES "hopper")
    list(APPEND cuda_archs 90)
  endif()

  if(cuda_archs)
    list(LENGTH cuda_archs num_archs)
    if(num_archs GREATER 1)
      # A CMake architecture list entry of "80" means to build both compute and sm. What
      # we want is for the newest arch only to build that way, while the rest build only
      # for sm.
      list(POP_BACK cuda_archs latest_arch)
      list(TRANSFORM cuda_archs APPEND "-real")
      list(APPEND cuda_archs ${latest_arch})
    else()
      list(TRANSFORM cuda_archs APPEND "-real")
    endif()
    set(${_LEGATE_DEST_VAR} ${cuda_archs} PARENT_SCOPE)
  endif()
endfunction()
