# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

def get_libpath(lib_base_name: str, full_lib_name: str) -> str: ...

LEGATE_ARCH: str

libpath: str

networks: list[str]

max_dim: int

max_fields: int

conduit: str

build_type: str

ON: bool

OFF: bool

use_cuda: bool

use_openmp: bool

legion_version: str

legion_git_branch: str

legion_git_repo: str

wheel_build: bool
