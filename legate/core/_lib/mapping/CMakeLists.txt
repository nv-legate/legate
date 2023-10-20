#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

set(cython_sources
    machine.pyx
    mapping.pyx
)

rapids_cython_create_modules(
  CXX
  ASSOCIATED_TARGETS legate_core
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES legate::core
)
