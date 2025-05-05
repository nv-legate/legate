# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ..type.types import Type
from ..utilities.unconstructable import Unconstructable

class TaskLocalBuffer(Unconstructable):
    @property
    def type(self) -> Type: ...
    @property
    def dim(self) -> int: ...
    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...
