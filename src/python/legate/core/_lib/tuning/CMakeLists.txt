#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "tuning")

set(cython_sources parallel_policy.pyx scope.pyx)

legate_create_cython_modules(SOURCES "${cython_sources}")
