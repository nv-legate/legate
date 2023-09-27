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

cmake_path(SET LEGATE_CORE_DIR NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../../")

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(${LEGATE_CORE_DIR}/cmake/legate_helper_functions.cmake)

legate_ensure_legate()

enable_testing()

file(GLOB main_SRC ${PROJECT_SOURCE_DIR}/main.cc)
file(GLOB integration_SRC ${PROJECT_SOURCE_DIR}/integration/*.cc)
file(GLOB tasks_SRC ${PROJECT_SOURCE_DIR}/integration/tasks/*.cc)
file(GLOB unit_SRC ${PROJECT_SOURCE_DIR}/unit/*.cc)

if(Legion_USE_CUDA)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

file(GLOB integration_GPU_SRC ${PROJECT_SOURCE_DIR}/integration/*.cu)
list(APPEND integration_SRC ${integration_GPU_SRC})

legate_include_rapids()

include(cmake/thirdparty/get_nccl.cmake)

endif()

add_executable(cpp_tests ${main_SRC} ${tasks_SRC} ${integration_SRC} ${unit_SRC})

target_link_libraries(cpp_tests PRIVATE legate::core PRIVATE GTest::gtest)
if(Legion_USE_CUDA)
target_link_libraries(cpp_tests PRIVATE NCCL::NCCL)
endif()

include(GoogleTest)
set_property(TARGET cpp_tests PROPERTY ENABLE_EXPORTS ON)
gtest_discover_tests(cpp_tests DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

include(CPack)
include(GNUInstallDirs)

install(TARGETS cpp_tests DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/cmake-install")
install(TARGETS cpp_tests DESTINATION ${CMAKE_INSTALL_BINDIR})
