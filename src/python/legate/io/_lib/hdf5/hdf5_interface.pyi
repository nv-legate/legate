# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from os import PathLike as os_PathLike
from pathlib import Path
from typing import TypeAlias

from ....core import LogicalStore
from ....core.data_interface import LogicalStoreLike

Pathlike: TypeAlias = str | os_PathLike[str] | Path

def from_file(path: Pathlike, dataset_name: str) -> LogicalStore: ...
def to_file(
    obj: LogicalStoreLike, path: object, dataset_name: str
) -> None: ...
