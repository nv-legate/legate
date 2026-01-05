# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from legate_printers.utils import (
    ArrayChildrenProvider,
    VectorChildrenProvider,
    get_type_str,
)

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

    def __init__(self, val: Value):
        self._val = val

        # get underlying storage value
        basic_var_type = gdb.types.get_basic_type(val["data_"].type)
        template_args = get_template_args(basic_var_type)
        variant_index = int(val["data_"]["_M_index"])
        storage_type = template_args[variant_index]
        storage_addr = val["data_"]["_M_u"]["_M_first"]["_M_storage"].address
        storage_value = storage_addr.cast(storage_type.pointer()).dereference()

        # get start and finish pointers and size of array
        if variant_index == 0:
            # cuda::std::inplace_vector
            basic_vec_type = gdb.types.get_basic_type(val.type)
            elem_type = get_template_args(basic_vec_type)[0]
            elem_ptr = storage_value["__elems_"].cast(elem_type.pointer())
            size = int(storage_value["__size_"])

            self._is_small = True
            self._children_provider = ArrayChildrenProvider(elem_ptr, size)
        else:
            # std::vector
            self._is_small = False
            self._children_provider = VectorChildrenProvider(storage_value)

    def children(self) -> Iterator[tuple[str, Value]]:
        """Return an iterator over the small-vector elements."""
        return self._children_provider.children()

    def to_string(self) -> str:
        """Return header string representation of the small-vector."""
        mode_str = "small" if self._is_small else "BIG"
        size_str = self._children_provider.size_str()
        return f"{get_type_str(self._val)} size={size_str} mode={mode_str}"

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
