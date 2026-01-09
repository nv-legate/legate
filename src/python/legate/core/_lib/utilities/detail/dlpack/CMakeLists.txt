#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "dlpack")

set(cython_sources dlpack.pyx from_dlpack.pyx to_dlpack.pyx)

legate_create_cython_modules(SOURCES "${cython_sources}")
