#=============================================================================
# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
