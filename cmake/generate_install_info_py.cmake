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

execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -E -DLEGATE_USE_PYTHON_CFFI
    -I "${CMAKE_CURRENT_LIST_DIR}/../src/core"
    -P "${CMAKE_CURRENT_LIST_DIR}/../src/core/legate_c.h"
  ECHO_ERROR_VARIABLE
  OUTPUT_VARIABLE header
  COMMAND_ERROR_IS_FATAL ANY
)

set(libpath "")
configure_file(
  "${LEGATE_CORE_DIR}/legate/install_info.py.in"
  "${LEGATE_CORE_DIR}/legate/install_info.py"
  @ONLY
)
