#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "runtime")

add_subdirectory(detail)

set(cython_sources exception_mode.pyx library.pyx resource.pyx runtime.pyx)

legate_create_cython_modules(SOURCES "${cython_sources}")
