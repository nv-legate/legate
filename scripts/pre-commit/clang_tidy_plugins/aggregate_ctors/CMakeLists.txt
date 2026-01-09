#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "aggregate_ctors")

legate_create_clang_tidy_plugin(aggregate_ctors SOURCES aggregate_ctors.cc)
