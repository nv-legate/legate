#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "task")

set(cython_sources task_context.pyx task_info.pyx task_signature.pyx variant_options.pyx
                   task_config.pyx)

legate_create_cython_modules(SOURCES "${cython_sources}")
