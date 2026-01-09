#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "detail")

set(cython_sources config.pyx)

legate_create_cython_modules(SOURCES "${cython_sources}")
