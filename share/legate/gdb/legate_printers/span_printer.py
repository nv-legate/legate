# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import gdb

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gdb import Value
    from gdb.printing import RegexpCollectionPrettyPrinter


class SpanPrinter(gdb.ValuePrinter):
    """Printer class for a legate::Span."""

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
        self._start = val["data_"]
        self._size = int(val["size_"])

    def children(self) -> Iterator[tuple[str, Value]]:
        """Return an iterator over the span elements."""
        return self._Iterator(self._start, self._start + self._size)

    def to_string(self) -> str:
        """Return header string representation of the span."""
        return f"{self._val.type.tag} size={self._size}"

    def display_hint(self) -> str:
        """Return display hint for the span."""
        return "array"


def register_printer(pp: RegexpCollectionPrettyPrinter) -> None:
    """Register the legate::Span printer."""
    pp.add_printer("Span", "^legate::Span<.*>$", SpanPrinter)
