# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from os import PathLike as os_PathLike
from pathlib import Path
from typing import TypeAlias

from ....core import LogicalArray

Pathlike: TypeAlias = str | os_PathLike[str] | Path

def from_file(path: Pathlike, dataset_name: str) -> LogicalArray: ...
