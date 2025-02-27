#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

cmake_minimum_required(VERSION 3.26.4)

project(legate-python VERSION 25.03.00 LANGUAGES CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(LEGATE_BUILD_PIP_WHEELS ON)

add_subdirectory(../../../../src legate-all)

set(rpaths "$ORIGIN/../../nvidia/nccl/lib")
set_property(TARGET legate PROPERTY INSTALL_RPATH ${rpaths} APPEND)
