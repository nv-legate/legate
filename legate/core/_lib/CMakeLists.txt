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

set(cython_sources context.pyx types.pyx)
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
