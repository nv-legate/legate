#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "task")

set(cython_sources decorator.pyx invoker.pyx py_task.pyx type.pyx util.pyx
                   python_task.pyx)

legate_create_cython_modules(SOURCES "${cython_sources}")
