#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(cpp_tests VERSION 0.1 LANGUAGES C CXX)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(legate_core REQUIRED)
find_package(cunumeric REQUIRED)

enable_testing()

file(GLOB main_SRC ${PROJECT_SOURCE_DIR}/main.cc)
file(GLOB hello_SRC ${PROJECT_SOURCE_DIR}/hello/*.cc)
file(GLOB io_SRC ${PROJECT_SOURCE_DIR}/io/*.cc)

add_executable(cpp_tests ${main_SRC} ${hello_SRC} ${io_SRC})

target_link_libraries(cpp_tests PRIVATE legate::core PRIVATE cunumeric::cunumeric PRIVATE GTest::gtest)

include(GoogleTest)
gtest_discover_tests(cpp_tests)

install(TARGETS cpp_tests DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/cmake-install")
