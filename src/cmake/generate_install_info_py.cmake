#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

set(libpath "")
configure_file("${LEGATE_CMAKE_DIR}/templates/install_info.py.in"
               "${LEGATE_DIR}/src/python/legate/install_info.py" @ONLY)
