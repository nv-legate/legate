# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, unique
from typing import cast

@unique
class StreamingMode(IntEnum):
    OFF = cast(int, ...)
    STRICT = cast(int, ...)
    RELAXED = cast(int, ...)

class ParallelPolicy:
    def __init__(
        self,
        *,
        streaming_mode: StreamingMode = ...,
        overdecompose_factor: int = 1,
    ) -> None: ...
    @property
    def streaming(self) -> bool: ...
    @property
    def streaming_mode(self) -> StreamingMode: ...
    @streaming_mode.setter
    def streaming_mode(self, mode: StreamingMode) -> bool: ...
    @property
    def overdecompose_factor(self) -> int: ...
    @overdecompose_factor.setter
    def overdecompose_factor(self, overdecompose_factor: int) -> int: ...
