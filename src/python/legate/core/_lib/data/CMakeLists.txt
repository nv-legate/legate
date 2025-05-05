#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "data")

set(cython_sources
    external_allocation.pyx inline_allocation.pyx logical_array.pyx logical_store.pyx
    physical_array.pyx physical_store.pyx scalar.pyx shape.pyx slice.pyx buffer.pyx)

legate_create_cython_modules(SOURCES "${cython_sources}")
