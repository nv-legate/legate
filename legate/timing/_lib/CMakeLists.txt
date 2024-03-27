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

# in case we ever need to support msvc
if(MSVC)
  message(FATAL_ERROR "Don't know the flag to silence all MSVC warnings")
endif()
# Silence all compiler warnings for cythonized C/C++ code. We could try and be smart about
# this, and only remove/silence the specific flags that cause warnings, but at the end of
# the day it will be an arms race we cannot win.
add_compile_options(-w)

set(cython_sources timing.pyx)
set(linked_libraries legate::core)

rapids_cython_create_modules(
  CXX
  ASSOCIATED_TARGETS legate_core
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
)

find_package (Python3 REQUIRED COMPONENTS Development)

foreach(target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  target_compile_features(${target} PRIVATE cxx_std_17)
  # Mark python headers as system (in case user has added warning flags, we don't want the
  # python headers to trigger them)
  target_include_directories(${target} SYSTEM PRIVATE ${Python3_INCLUDE_DIRS})
endforeach()
