#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

# ------------------------------------------------------------------------------#
# Architecture
# ------------------------------------------------------------------------------#
if(BUILD_MARCH AND BUILD_MCPU)
  message(FATAL_ERROR "BUILD_MARCH and BUILD_MCPU are incompatible")
endif()

include("${LEGATE_CMAKE_DIR}/Modules/utilities.cmake")

function(set_cpu_arch_flags_impl NAME flags_out_var success_var)
  string(TOLOWER "${NAME}" name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "${name}")

  set(${success_var} FALSE)
  set_parent_scope(${success_var})
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
  legate_check_compiler_flag(CXX "${flag}" ${success_var})

  if((NOT ${success_var}) AND BUILD_${NAME})
    message(FATAL_ERROR "The flag ${flag} is not supported by the compiler")
  endif()
  set_parent_scope(${success_var})
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

  # Add flags for Power architectures - only check on appropriate platforms
  # CMAKE_SYSTEM_PROCESSOR might be ppc64le, powerpc64le, etc. for PowerPC
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc|powerpc|power)")
    foreach(flag "-maltivec" "-mabi=altivec" "-mvsx")
      legate_check_compiler_flag(CXX "${flag}" success)
      if(success)
        list(APPEND flags "${flag}")
      endif()
    endforeach()
  endif()

  list(APPEND ${out_var} ${flags})
  set_parent_scope(${out_var})
endfunction()
