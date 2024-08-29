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

list(APPEND CMAKE_MESSAGE_CONTEXT "cpp")

include(${LEGATE_CORE_DIR}/cmake/Modules/debug_symbols.cmake)

legate_core_find_or_configure(PACKAGE google_benchmark)

function(configure_benchmark)
  set(options)
  set(one_value TARGET)
  set(multi_value SOURCES)
  cmake_parse_arguments(_LEGATE_BM "${options}" "${one_value}" "${multi_value}" ${ARGN})

  add_executable(${_LEGATE_BM_TARGET} ${_LEGATE_BM_SOURCES})

  set_target_properties(${_LEGATE_BM_TARGET}
                        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                   "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}>"
                                   BUILD_RPATH_USE_ORIGIN TRUE
                                   # Don't use CMAKE_INSTALL_LIBDIR since we want the
                                   # rpath to be relative
                                   INSTALL_RPATH
                                   "${LEGATE_CORE_PLATFORM_RPATH_ORIGIN}/../lib"
                                   INSTALL_RPATH_USE_LINK_PATH TRUE
                                   CXX_STANDARD ${CMAKE_CXX_STANDARD}
                                   CXX_STANDARD_REQUIRED ON
                                   # For std:: support of __int128_t. Can be removed once
                                   # using cuda::std
                                   CXX_EXTENSIONS ON)

  target_link_libraries(${_LEGATE_BM_TARGET} PRIVATE legate::core benchmark::benchmark)

  if(Legion_USE_CUDA)
    set_target_properties(${_LEGATE_BM_TARGET}
                          PROPERTIES CUDA_ARCHITECTURES ${Legion_CUDA_ARCH}
                                     CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
                                     CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(${_LEGATE_BM_TARGET} PRIVATE NCCL::NCCL)
  endif()

  legate_core_add_target_compile_option(${_LEGATE_BM_TARGET} CXX PRIVATE
                                        legate_core_CXX_OPTIONS)
  legate_core_add_target_compile_option(${_LEGATE_BM_TARGET} CUDA PRIVATE
                                        legate_core_CUDA_OPTIONS)

  legate_core_add_target_compile_option(${_LEGATE_BM_TARGET} CXX PRIVATE
                                        legate_core_CXX_FLAGS)
  legate_core_add_target_compile_option(${_LEGATE_BM_TARGET} CUDA PRIVATE
                                        legate_core_CUDA_FLAGS)

  legate_core_add_target_link_option(${_LEGATE_BM_TARGET} PRIVATE
                                     legate_core_LINKER_FLAGS)

  legate_core_debug_syms(${_LEGATE_BM_TARGET} INSTALL_DIR ${CMAKE_INSTALL_BINDIR})

  list(APPEND legate_core_TIDY_SOURCES ${_LEGATE_TEST_SOURCES})
  set(legate_core_TIDY_SOURCES ${legate_core_TIDY_SOURCES} PARENT_SCOPE)
endfunction()

include(GNUInstallDirs)

configure_benchmark(TARGET inline_launch SOURCES inline_launch.cc)

# propagate to main CMakeLists.txt
set(legate_core_TIDY_SOURCES ${legate_core_TIDY_SOURCES} PARENT_SCOPE)
