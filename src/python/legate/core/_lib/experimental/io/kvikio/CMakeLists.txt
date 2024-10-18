#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "kvikio")

# The reason we call it kvikio_interface.pyx instead of interface.pyx (to match the C++
# files) is because you get:
#
# CMake Error at
# /opt/homebrew/Cellar/cmake/3.30.4/share/cmake/Modules/FindPython/Support.cmake:4239
# (add_library): add_library cannot create target "interface" because another target with
# the same name already exists.  The existing target is a module library created in source
# directory "/path/to/src/python/legate/_lib/io/kvikio". See documentation for policy
# CMP0002 for more details.
set(cython_sources kvikio_interface.pyx)

legate_create_cython_modules(SOURCES "${cython_sources}")
