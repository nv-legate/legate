#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function(legate_export_aedifix_post_config)
  list(APPEND CMAKE_MESSAGE_CONTEXT "export_aedifix_post_config")

  if(NOT AEDIFIX)
    message(VERBOSE "AEDIFIX not defined, no need to emit export config")
    return()
  endif()

  set(data "{}")
  foreach(var IN LISTS AEDIFIX_EXPORT_VARIABLES)
    string(JSON data SET "${data}" "${var}" "\"${${var}}\"")
  endforeach()
  message(VERBOSE "Wrote aedifix export config to ${AEDIFIX_EXPORT_CONFIG_PATH}")
  file(WRITE "${AEDIFIX_EXPORT_CONFIG_PATH}" "${data}")
endfunction()
