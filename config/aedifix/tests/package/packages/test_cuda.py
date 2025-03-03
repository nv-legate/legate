# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import pytest

from ....package.packages.cuda import CudaArchAction

ARCH_STR: tuple[tuple[str, list[str]], ...] = (
    ("", []),
    (",,", []),
    ("70", ["70"]),
    ("70,80", ["70", "80"]),
    ("ampere", ["80"]),
    ("turing,hopper", ["75", "90"]),
    ("volta,60,all-major", ["70", "60", "all-major"]),
    ("60,,80", ["60", "80"]),
)


class TestCudaArchAction:
    @pytest.mark.parametrize(("argv", "expected"), ARCH_STR)
    def test_map_cuda_arch_names(self, argv: str, expected: list[str]) -> None:
        ret = CudaArchAction.map_cuda_arch_names(argv)
        assert ret == expected


if __name__ == "__main__":
    sys.exit(pytest.main())
