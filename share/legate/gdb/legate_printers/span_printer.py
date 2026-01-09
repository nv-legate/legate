# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from legate_printers.utils import ArrayChildrenProvider, get_type_str

import gdb

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gdb import Value
    from gdb.printing import RegexpCollectionPrettyPrinter


class SpanPrinter(gdb.ValuePrinter):
    """Printer class for a legate::Span."""

    def __init__(self, val: Value):
        self._val = val
        self._children_provider = ArrayChildrenProvider(
            val["__data_"], int(val["__size_"])
        )

    def children(self) -> Iterator[tuple[str, Value]]:
        """Return an iterator over the span elements."""
        return self._children_provider.children()

    def to_string(self) -> str:
        """Return header string representation of the span."""
        size_str = self._children_provider.size_str()
        return f"{get_type_str(self._val)} size={size_str}"

    def display_hint(self) -> str:
        """Return display hint for the span."""
        return "array"


def register_printer(pp: RegexpCollectionPrettyPrinter) -> None:
    """Register the legate::Span printer."""
    pp.add_printer("Span", "^cuda::std::.*span<.*>$", SpanPrinter)
