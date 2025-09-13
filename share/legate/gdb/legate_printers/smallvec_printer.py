# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import gdb
from gdb import Type, Value

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gdb.printing import RegexpCollectionPrettyPrinter


def get_template_args(gdb_type: Type) -> list[Type]:
    """Given a GDB type, return a list of its template arguments."""
    n = 0
    template_args = []
    while True:
        try:
            template_args.append(gdb_type.template_argument(n))
        except Exception as _:
            return template_args
        n += 1


class SmallVectorPrinter(gdb.ValuePrinter):
    """Printer class for a legate::detail::SmallVector."""

    class _Iterator:
        def __init__(self, start: Value, finish: Value):
            self._item = start
            self._finish = finish
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self) -> tuple[str, Value]:
            if self._item == self._finish:
                raise StopIteration
            elt = self._item.dereference()
            value_idx = self._index
            self._item = self._item + 1
            self._index += 1
            return (f"[{value_idx}]", elt)

    def __init__(self, val: Value):
        self._val = val

        # get underlying storage value
        template_args = get_template_args(val["data_"].type)
        variant_index = int(val["data_"]["_M_index"])
        storage_type = template_args[variant_index]
        storage_addr = val["data_"]["_M_u"]["_M_first"]["_M_storage"].address
        storage_value = storage_addr.cast(storage_type.pointer()).dereference()

        # get start and finish pointers and size of array
        if variant_index == 0:
            # cuda::std::inplace_vector
            elems = storage_value["__elems_"]
            elem_type = elems.type.target()
            size = int(storage_value["__size_"])

            # get pointer to the start/end of the array
            self._start = elems.cast(elem_type.pointer())
            self._finish = self._start + size
            self._size = size
            self._is_small = True
            self._provide_children = True
        else:
            # std::vector
            self._is_small = False

            # (std::vector<bool> may be optimized into a bitvector for
            # which this implementation is not suited for handling.
            # Don't print vector elements in this case.)
            is_bool = (
                self._val.type.template_argument(0).code == gdb.TYPE_CODE_BOOL
            )
            self._provide_children = not is_bool
            if self._provide_children:
                self._start = storage_value["_M_impl"]["_M_start"]
                self._finish = storage_value["_M_impl"]["_M_finish"]
                self._size = int(self._finish - self._start)
            else:
                self._size = "bit-vector size=?"

    def children(self) -> Iterator[tuple[str, Value]]:
        """Return an iterator over the small-vector elements.

        If the vector is not providing children, return an empty iterator.
        """
        if self._provide_children:
            return self._Iterator(self._start, self._finish)
        # provide empty iterator if not providing children
        return iter([])

    def to_string(self) -> str:
        """Return header string representation of the small-vector."""
        mode_str = "small" if self._is_small else "BIG"
        return f"{self._val.type.tag} of size={self._size} mode={mode_str}"

    def display_hint(self) -> str:
        """Return display hint for the small-vector."""
        return "array"


def register_printer(pp: RegexpCollectionPrettyPrinter) -> None:
    """Register the legate::detail::SmallVector printer."""
    pp.add_printer(
        "SmallVector",
        "^legate::detail::SmallVector<.*,.*>$",
        SmallVectorPrinter,
    )
