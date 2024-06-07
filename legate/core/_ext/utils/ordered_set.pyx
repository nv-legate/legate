# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Iterable
from typing import TypeVar


cdef object T = TypeVar("T")

cdef class OrderedSet:
    """
    A set() variant whose iterator returns elements in insertion order.

    The implementation of this class piggybacks off of the corresponding
    iteration order guarantee for dict(), starting with Python 3.7. This is
    useful for guaranteeing symmetric execution of algorithms on different
    shards in a replicated context.
    """

    def __init__(self, copy_from: Iterable[T] | None = None) -> None:
        self._dict = {}
        if copy_from is not None:
            for obj in copy_from:
                self.add(obj)

    cpdef void add(self, object obj):
        self._dict[obj] = None

    cpdef void update(self, object iterable):
        for obj in iterable:
            self.add(obj)

    cpdef void discard(self, object obj):
        self._dict.pop(obj, None)

    def __len__(self) -> int:
        return len(self._dict)

    def __contains__(self, obj: Any) -> bool:
        return obj in self._dict

    def __iter__(self) -> Iterator[T]:
        return iter(self._dict)

    cpdef OrderedSet remove_all(self, OrderedSet other):
        return OrderedSet([obj for obj in self if obj not in other])
