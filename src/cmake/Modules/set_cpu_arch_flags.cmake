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

include_guard(GLOBAL)

# ------------------------------------------------------------------------------#
# Architecture
# ------------------------------------------------------------------------------#
if(BUILD_MARCH AND BUILD_MCPU)
  message(FATAL_ERROR "BUILD_MARCH and BUILD_MCPU are incompatible")
endif()

include(CheckCXXCompilerFlag)

function(set_cpu_arch_flags_impl NAME flags_out_var success_var)
  string(TOLOWER "${NAME}" name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "${name}")

  set(${success_var} FALSE PARENT_SCOPE)
  if(BUILD_${NAME})
    message(VERBOSE "Using BUILD_${NAME}=${BUILD_${NAME}} (user-defined)")
  elseif(NOT DEFINED BUILD_${NAME})
    set(BUILD_${NAME} "native")
    message(VERBOSE "Using BUILD_${NAME}=${BUILD_${NAME}} (default)")
  else()
    message(VERBOSE "Skipping ${name} check due to BUILD_${NAME}=${BUILD_${NAME}}")
    return()
  endif()

  set(flag "-${name}=${BUILD_${NAME}}")
  check_cxx_compiler_flag("${flag}" COMPILER_SUPPORTS_${NAME})
  if(COMPILER_SUPPORTS_${NAME})
    list(APPEND ${flags_out_var} "${flag}")
    set(${flags_out_var} "${${flags_out_var}}" PARENT_SCOPE)
    set(${success_var} TRUE PARENT_SCOPE)
  elseif(BUILD_${NAME})
    message(FATAL_ERROR "The flag ${flag} is not supported by the compiler")
  endif()
endfunction()

function(set_cpu_arch_flags out_var)
  list(APPEND CMAKE_MESSAGE_CONTEXT "set_cpu_arch_flags")

  set(flags "")
  set(success FALSE)
  # Try -march first. On platforms that don't support it, GCC will issue a hard error, so
  # we'll know not to use it. Default is "native", but explicitly setting BUILD_MARCH=""
  # disables use of the flag
  set_cpu_arch_flags_impl(MARCH flags success)

  # Try -mcpu. We do this second because it is deprecated on x86, but GCC won't issue a
  # hard error, so we can't tell if it worked or not.
  if(NOT success)
    set_cpu_arch_flags_impl(MCPU flags success)
  endif()

  # Add flags for Power architectures
  check_cxx_compiler_flag("-maltivec -Werror" COMPILER_SUPPORTS_MALTIVEC)
  if(COMPILER_SUPPORTS_MALTIVEC)
    list(APPEND flags "-maltivec")
  endif()
  check_cxx_compiler_flag("-mabi=altivec -Werror" COMPILER_SUPPORTS_MABI_ALTIVEC)
  if(COMPILER_SUPPORTS_MABI_ALTIVEC)
    list(APPEND flags "-mabi=altivec")
  endif()
  check_cxx_compiler_flag("-mvsx -Werror" COMPILER_SUPPORTS_MVSX)
  if(COMPILER_SUPPORTS_MVSX)
    list(APPEND flags "-mvsx")
  endif()

  list(APPEND ${out_var} ${flags})
  set(${out_var} "${${out_var}}" PARENT_SCOPE)
endfunction()
