#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_nccl)
  list(APPEND CMAKE_MESSAGE_CONTEXT "nccl")

  if(TARGET NCCL::NCCL)
    return()
  endif()

  rapids_find_generate_module(NCCL HEADER_NAMES nccl.h LIBRARY_NAMES nccl)
  # Currently NCCL has no CMake build-system so we require it built and installed on the
  # machine already
  rapids_find_package(NCCL)
  if(TARGET NCCL::NCCL)
    return()
  endif()

  # If the user has installed NCCL to a system location, then CMake might not find their
  # NCCL because it will be under /usr/lib/<whatever>. CMake will, however, search under
  # there if CMAKE_LIBRARY_ARCHITECTURE is set, so we need to ensure that it is.
  if(CMAKE_LIBRARY_ARCHITECTURE)
    # In this case, the below find_package() will fail (we are doing the same thing as
    # above without changing anything), so better to fail with a more useful error message
    # instead.
    message(VERBOSE "CMAKE_LIBRARY_ARCHITECTURE is set: ${CMAKE_LIBRARY_ARCHITECTURE}")
    message(FATAL_ERROR "Could not find NCCL on your system. It's possible that you have it installed in a location that CMake does not know to search for automatically. Please report this case to Legate maintainers."
    )
  endif()

  message(VERBOSE "CMAKE_LIBRARY_ARCHITECTURE is unset, attempting to deduce it")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(VERBOSE "linux system detected")
    foreach(arch_dir IN ITEMS x86_64-linux-gnu aarch64-linux-gnu)
      if(EXISTS "/usr/lib/${arch_dir}")
        set(CMAKE_LIBRARY_ARCHITECTURE "${arch_dir}")
        break()
      endif()
    endforeach()
  endif()

  if(NOT CMAKE_LIBRARY_ARCHITECTURE)
    message(FATAL_ERROR "Could not auto-deduce CMAKE_LIBRARY_ARCHITECTURE while trying to locate NCCL. It's possible that you have NCCL installed in a location that CMake does not know to search for automatically (which we tried to remedy by deducing CMAKE_LIBRARY_ARCHITECTURE). Please report this case to Legate maintainers."
    )
  endif()

  message(VERBOSE "CMAKE_LIBRARY_ARCHITECTURE is set: ${CMAKE_LIBRARY_ARCHITECTURE}")
  rapids_find_package(NCCL REQUIRED)
endfunction()
