# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import gdb

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gdb import Value


class StartEndIterator:
    """Iterator class over array elems given start and finish pointers."""

    def __init__(self, start: Value, finish: Value):
        self._item = start
        self._finish = finish
        self._index = 0

    def __iter__(self):
        """Return the iterator itself."""
        return self

    def __next__(self) -> tuple[str, Value]:
        """Return the next element in the iterator.

        Returns
        -------
            (name, value): tuple[str, Value]
                The name and value of the next child in the iterator.
        """
        if self._item == self._finish:
            raise StopIteration
        elt = self._item.dereference()
        value_idx = self._index
        self._item = self._item + 1
        self._index += 1
        return (f"[{value_idx}]", elt)


class ArrayChildrenProvider:
    """Class for providing iterable over array elements."""

    def __init__(self, address: Value, size: int):
        self._address = address
        self._size = size

    def children(self) -> Iterator[tuple[str, Value]]:
        """Return an iterator over the array elements."""
        return StartEndIterator(self._address, self._address + self._size)

    def size_str(self) -> str:
        """Return string representation of the array size."""
        return f"{self._size}"


class VectorChildrenProvider:
    """Class for providing iterable over std::vector elements.

    If the vector is a bool vector, this class provides an
    empty iterable to avoid assuming if the vector is an optimized bitvector.
    """

    def __init__(self, vector: Value):
        self._vector = vector

        vec_type = gdb.types.get_basic_type(self._vector.type)
        is_bool = vec_type.template_argument(0).code == gdb.TYPE_CODE_BOOL
        self._provide_children = not is_bool

    def children(self) -> Iterator[tuple[str, Value]]:
        """Return an iterator over the vector elements.

        If the vector is a bool vector, returns an empty iterable.
        """
        if self._provide_children:
            start = self._vector["_M_impl"]["_M_start"]
            finish = self._vector["_M_impl"]["_M_finish"]
            return StartEndIterator(start, finish)
        return iter([])

    def size_str(self) -> str:
        """Return string representation of the vector size."""
        if self._provide_children:
            start = self._vector["_M_impl"]["_M_start"]
            finish = self._vector["_M_impl"]["_M_finish"]
            return f"{int(finish - start)}"
        return "bit-vector ?"


def is_reference(val: Value) -> bool:
    """Return True if the value is a reference."""
    try:
        _ = val.referenced_value()
    except Exception:
        return False
    else:
        return True


def get_type_str(val: Value) -> str:
    """Return the string representation of the type."""
    if is_reference(val):
        # GDB Python API returns None for val.type.tag when
        # val.type is a reference for some reason. As a result,
        # we remove all qualifiers from the type to get a
        # proper string and add back in the reference `&`
        return f"{gdb.types.get_basic_type(val.type).tag}&"
    return val.type.tag
