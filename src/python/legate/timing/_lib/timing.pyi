# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from ...core._lib.utilities.unconstructable import Unconstructable

class PyTime(Unconstructable):
    @staticmethod
    def measure_microseconds() -> PyTime: ...
    @staticmethod
    def measure_nanoseconds() -> PyTime: ...
    def value(self) -> int: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __add__(self, rhs: int | PyTime) -> int: ...
    def __radd__(self, lhs: int | PyTime) -> int: ...
    def __sub__(self, rhs: int | PyTime) -> int: ...
    def __rsub__(self, lhs: int | PyTime) -> int: ...
    def __mul__(self, rhs: int | PyTime) -> int: ...
    def __rmul__(self, lhs: int | PyTime) -> int: ...
    def __div__(self, rhs: int | PyTime) -> float: ...
    def __rdiv__(self, lhs: int | PyTime) -> float: ...

def time(units: str) -> PyTime: ...
