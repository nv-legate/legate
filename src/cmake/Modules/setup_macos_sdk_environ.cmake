#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(_set_macosx_deployment_target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "macosx_deployment_target")

  if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    message(
      VERBOSE
      "CMAKE_OSX_DEPLOYMENT_TARGET already set: ${CMAKE_OSX_DEPLOYMENT_TARGET}. "
      "Letting CMake detect defaults from it."
    )
    return()
  endif()

  if(DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
    message(
      VERBOSE
      "MACOSX_DEPLOYMENT_TARGET already set: $ENV{MACOSX_DEPLOYMENT_TARGET}, "
      "letting CMake detect defaults from it."
    )
    return()
  endif()

  find_program(LEGATE_SW_VERS sw_vers)
  if(NOT LEGATE_SW_VERS)
    message(
      VERBOSE
      "Failed to detect macOS version, cannot set CMAKE_OSX_DEPLOYMENT_TARGET. "
      "Letting CMake detect defaults for it."
    )
    return()
  endif()

  execute_process(
    COMMAND ${LEGATE_SW_VERS} --productVersion
    COMMAND_ERROR_IS_FATAL ANY
    OUTPUT_VARIABLE macos_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(ENV{MACOSX_DEPLOYMENT_TARGET} "${macos_version}")
  # If the docstring for CMAKE_OSX_DEPLOYMENT_TARGET below looks a little verbose, it's
  # because it exactly matches the one that CMake uses
  set(
    CMAKE_OSX_DEPLOYMENT_TARGET
    "${macos_version}"
    CACHE STRING
    "Minimum OS X version to target for deployment (at runtime); newer APIs weak linked. Set to empty string for default value."
  )
  message(VERBOSE "Set CMAKE_OSX_DEPLOYMENT_TARGET=${macos_version}")
endfunction()

function(_set_macosx_sdkroot)
  list(APPEND CMAKE_MESSAGE_CONTEXT "sdkroot")

  if(DEFINED CMAKE_OSX_SYSROOT)
    message(
      VERBOSE
      "CMAKE_OSX_SYSROOT already set: ${CMAKE_OSX_SYSROOT}. "
      "Letting CMake detect defaults from it."
    )
    return()
  endif()

  if(DEFINED ENV{SDKROOT})
    message(
      VERBOSE
      "SDKROOT already set: $ENV{SDKROOT}, letting CMake detect defaults from it."
    )
    return()
  endif()

  # See https://cmake.org/cmake/help/latest/release/4.0.html#other-changes, and
  # https://cmake.org/cmake/help/latest/variable/CMAKE_OSX_SYSROOT.html#variable:CMAKE_OSX_SYSROOT

  find_program(LEGATE_XCRUN xcrun)
  if(NOT LEGATE_XCRUN)
    message(
      VERBOSE
      "Failed to detect current SDK root directory, cannot set CMAKE_OSX_SYSROOT. "
      "Letting CMake detect defaults for it."
    )
    return()
  endif()

  execute_process(
    COMMAND ${LEGATE_XCRUN} --sdk macosx --show-sdk-path
    COMMAND_ERROR_IS_FATAL ANY
    OUTPUT_VARIABLE sdk_root
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(ENV{SDKROOT} "${sdk_root}")
  # If the docstring for CMAKE_OSX_SYSROOT below looks a little verbose, it's because it
  # exactly matches the one that CMake uses
  set(
    CMAKE_OSX_SYSROOT
    "${sdk_root}"
    CACHE STRING
    "The product will be built against the headers and libraries located inside the indicated SDK."
  )
  message(VERBOSE "Set CMAKE_OSX_SYSROOT=${sdk_root}")
endfunction()

function(legate_setup_macos_sdk_environ)
  list(APPEND CMAKE_MESSAGE_CONTEXT "setup_macos_sdk_environ")

  _set_macosx_deployment_target()
  _set_macosx_sdkroot()
endfunction()
