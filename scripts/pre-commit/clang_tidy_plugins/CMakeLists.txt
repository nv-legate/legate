#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.26.4)

list(APPEND CMAKE_MESSAGE_CONTEXT "clang_tidy_plugins")

# LLVM requires C to be enabled
project(clang_tidy_plugins VERSION 1.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(BUILD_SHARED_LIBS "Build Shared Libs" ON)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile commands" ON)
option(CMAKE_CXX_STANDARD_REQUIRED "Force C++ standard" ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG HINTS "${LLVM_CMAKE_DIR}/..")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(LEGATE_LLVM_LIBS core)

define_property(GLOBAL PROPERTY LEGATE_TIDY_PLUGINS)

function(legate_create_clang_tidy_plugin TARGET)
  list(APPEND CMAKE_MESSAGE_CONTEXT "create_clang_tidy_plugin")

  set(options)
  set(one_value_args)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(
    _LEGATE
    "${options}"
    "${one_value_args}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  add_library("${TARGET}" MODULE "${_LEGATE_SOURCES}")

  target_include_directories(
    "${TARGET}"
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS}
  )
  target_compile_definitions("${TARGET}" PRIVATE ${LLVM_DEFINITIONS_LIST})
  target_link_libraries("${TARGET}" PRIVATE ${LEGATE_LLVM_LIBS} clangTidy)

  set_property(GLOBAL APPEND PROPERTY LEGATE_TIDY_PLUGINS "${TARGET}")
endfunction()

add_subdirectory(aggregate_ctors)

get_property(plugins GLOBAL PROPERTY LEGATE_TIDY_PLUGINS)
set(
  LEGATE_TIDY_PLUGINS
  "${plugins}"
  CACHE STRING
  "List of configured clang-tidy plugins"
  FORCE
)
set(LEGATE_TIDY_PLUGINS "${plugins}" PARENT_SCOPE)
