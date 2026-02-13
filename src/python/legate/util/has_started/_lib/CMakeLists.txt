#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "_lib")

set(cython_sources has_started.pyx)

legate_create_cython_modules(SOURCES "${cython_sources}")
