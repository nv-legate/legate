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

if(POLICY CMP0074)
  # find_package() uses <PackageName>_ROOT variables
  # https://cmake.org/cmake/help/latest/policy/CMP0074.html#policy:CMP0074
  cmake_policy(SET CMP0074 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
endif()

if(POLICY CMP0060)
  # Link libraries by full path even in implicit directories
  # https://cmake.org/cmake/help/latest/policy/CMP0060.html#policy:CMP0060
  cmake_policy(SET CMP0060 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0060 NEW)
endif()

if(POLICY CMP0077)
  # option() honors normal variables
  # https://cmake.org/cmake/help/latest/policy/CMP0077.html
  cmake_policy(SET CMP0077 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
endif()

if(POLICY CMP0096)
  # The project() command preserves leading zeros in version components
  # https://cmake.org/cmake/help/latest/policy/CMP0096.html
  cmake_policy(SET CMP0096 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0096 NEW)
endif()

if(POLICY CMP0126)
  # make set(CACHE) command not remove normal variable of the same name from the current scope
  # https://cmake.org/cmake/help/latest/policy/CMP0126.html
  cmake_policy(SET CMP0126 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
endif()

if(POLICY CMP0135)
  # make the timestamps of ExternalProject_ADD match the download time
  # https://cmake.org/cmake/help/latest/policy/CMP0135.html
  cmake_policy(SET CMP0135 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

if(POLICY CMP0067)
  # Honor language standard in try_compile() source-file signature.
  # https://cmake.org/cmake/help/latest/policy/CMP0067.html
  cmake_policy(SET CMP0067 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0067 NEW)
endif()
