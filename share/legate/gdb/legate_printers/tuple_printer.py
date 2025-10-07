# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from legate_printers.utils import VectorChildrenProvider, get_type_str

import gdb

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gdb import Value
    from gdb.printing import RegexpCollectionPrettyPrinter


class TuplePrinter(gdb.ValuePrinter):
    """Printer class for a legate::tuple."""

    def __init__(self, val: Value):
        self._val = val
        self._children_provider = VectorChildrenProvider(val["data_"])

    def children(self) -> Iterator[tuple[str, Value]]:
        """Return an iterator over the tuple elements."""
        return self._children_provider.children()

    def to_string(self) -> str:
        """Return header string representation of the tuple."""
        size_str = self._children_provider.size_str()
        return f"{get_type_str(self._val)} size={size_str}"

    def display_hint(self) -> str:
        """Return display hint for the tuple."""
        return "array"


def register_printer(pp: RegexpCollectionPrettyPrinter) -> None:
    """Register the legate::tuple printer."""
    pp.add_printer("Tuple", "^legate::tuple<.*>$", TuplePrinter)
