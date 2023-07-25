# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

""" """

from .. import install_info as info


def print_build_info() -> None:
    print(
        f"""Legate build configuration:
  build_type : {info.build_type}
  use_openmp : {info.use_openmp}
  use_cuda   : {info.use_cuda}
  networks   : {','.join(info.networks) if info.networks else ''}
  conduit    : {info.conduit}
"""
    )
