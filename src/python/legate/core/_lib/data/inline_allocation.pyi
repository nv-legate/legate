# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

class InlineAllocation:
    @property
    def ptr(self) -> int: ...
    @property
    def strides(self) -> tuple[int, ...]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...
