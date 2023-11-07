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

include_guard(GLOBAL)

macro(legate_core_export_compile_commands)
  if (NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  endif()

  if (CMAKE_EXPORT_COMPILE_COMMANDS)
    message("-- Symlinking compile_commands.json to root directory")
    file(
      CREATE_LINK
      ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
      ${CMAKE_CURRENT_LIST_DIR}/compile_commands.json
      SYMBOLIC
    )
  endif()
endmacro()
