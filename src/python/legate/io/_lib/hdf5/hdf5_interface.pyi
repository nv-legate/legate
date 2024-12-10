# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from os import PathLike as os_PathLike
from pathlib import Path
from typing import TypeAlias

from ....core import LogicalArray

Pathlike: TypeAlias = str | os_PathLike[str] | Path

def from_file(path: Pathlike, dataset_name: str) -> LogicalArray: ...
