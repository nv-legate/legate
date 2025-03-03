# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from legate.tester.test_system import TestSystem
from legate.util.types import CPUInfo, GPUInfo


class FakeSystem(TestSystem):
    def __init__(
        self, cpus: int = 6, gpus: int = 6, fbmem: int = 6 << 32, **kwargs: Any
    ) -> None:
        self._cpus = cpus
        self._gpus = gpus
        self._fbmem = fbmem
        super().__init__(**kwargs)

    @property
    def cpus(self) -> tuple[CPUInfo, ...]:
        return tuple(CPUInfo((i,)) for i in range(self._cpus))

    @property
    def gpus(self) -> tuple[GPUInfo, ...]:
        return tuple(GPUInfo(i, self._fbmem) for i in range(self._gpus))
