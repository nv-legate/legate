# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
