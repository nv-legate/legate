# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Protocol

from typing_extensions import CapsuleType

from ....data.logical_store import LogicalStore

class SupportsDLPack(Protocol):
    def __dlpack__(self) -> CapsuleType: ...

def from_dlpack(
    x: SupportsDLPack,
    /,
    *,
    device: tuple[int | Enum, int] | None = None,
    copy: bool | None = None,
) -> LogicalStore: ...
