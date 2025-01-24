# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int64_t
from libcpp.utility cimport move as std_move

from ...core._lib.utilities.unconstructable cimport Unconstructable


cdef class PyTime(Unconstructable):
    @staticmethod
    def measure_microseconds() -> PyTime:
        cdef PyTime result = PyTime.__new__(PyTime)
        with nogil:
            result._time = std_move(measure_microseconds())
        return result

    @staticmethod
    def measure_nanoseconds() -> PyTime:
        cdef PyTime result = PyTime.__new__(PyTime)
        with nogil:
            result._time = std_move(measure_nanoseconds())
        return result

    cpdef int64_t value(self):
        with nogil:
            return self._time.value()

    def __int__(self) -> int:
        return self.value()

    def __str__(self) -> str:
        return str(self.value())

    def __float__(self) -> float:
        return float(self.value())

    def __add__(self, rhs: int | PyTime) -> int:
        return self.value() + int(rhs)

    def __radd__(self, lhs: int | PyTime) -> int:
        return int(lhs) + self.value()

    def __sub__(self, rhs: int | PyTime) -> int:
        return self.value() - int(rhs)

    def __rsub__(self, lhs: int | PyTime) -> int:
        return int(lhs) - self.value()

    def __mul__(self, rhs: int | PyTime) -> int:
        return self.value() * int(rhs)

    def __rmul__(self, lhs: int | PyTime) -> int:
        return int(lhs) * self.value()

    def __div__(self, rhs: int | PyTime) -> float:
        return self.value() / int(rhs)

    def __rdiv__(self, lhs: int | PyTime) -> float:
        return int(lhs) / self.value()


cpdef PyTime time(str units = "us"):
    if units == "us":
        return PyTime.measure_microseconds()
    if units == "ns":
        return PyTime.measure_nanoseconds()
    raise ValueError("time units must be 'us' or 'ns'")
