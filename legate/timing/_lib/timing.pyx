# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libcpp.utility cimport move

from typing import Union

import cython

from legate.core import get_legate_runtime


cdef class PyTime:
    @staticmethod
    def measure_microseconds() -> PyTime:
        cdef PyTime result = PyTime.__new__(PyTime)
        result._time = move(measure_microseconds())
        return result

    @staticmethod
    def measure_nanoseconds() -> PyTime:
        cdef PyTime result = PyTime.__new__(PyTime)
        result._time = move(measure_nanoseconds())
        return result

    def value(self) -> int:
        return self._time.value()

    def __int__(self) -> int:
        return self.value()

    def __str__(self) -> str:
        return str(self.value())

    def __float__(self) -> float:
        return float(self.value())

    def __add__(self, rhs: Union[int, PyTime]) -> int:
        return self.value() + int(rhs)

    def __radd__(self, lhs: Union[int, PyTime]) -> int:
        return int(lhs) + self.value()

    def __sub__(self, rhs: Union[int, PyTime]) -> int:
        return self.value() - int(rhs)

    def __rsub__(self, lhs: Union[int, PyTime]) -> int:
        return int(lhs) - self.value()

    def __mul__(self, rhs: Union[int, PyTime]) -> int:
        return self.value() * int(rhs)

    def __rmul__(self, lhs: Union[int, PyTime]) -> int:
        return int(lhs) * self.value()

    def __div__(self, rhs: Union[int, PyTime]) -> float:
        return self.value() / int(rhs)

    def __rdiv__(self, lhs: Union[int, PyTime]) -> float:
        return int(lhs) / self.value()


def time(units: str = "us") -> PyTime:
    # Issue a Legion execution fence and then perform a timing operation
    # immediately after it
    get_legate_runtime().issue_execution_fence()
    if units == "us":
        return PyTime.measure_microseconds()
    elif units == "ns":
        return PyTime.measure_nanoseconds()
    else:
        raise ValueError("time units must be 'us' or 'ns'")
