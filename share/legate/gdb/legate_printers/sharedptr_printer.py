# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from legate_printers.utils import get_type_str

import gdb

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gdb import Value
    from gdb.printing import RegexpCollectionPrettyPrinter

SHARED_PTR_REGEX = re.compile(r"^legate::SharedPtr<.*>$")


def internal_ptr_string(val: Value) -> str:
    """Return a string representation of the internal pointer."""
    ctrl_block = val["ctrl_"]

    strong_refs = ctrl_block["strong_refs_"]["_M_i"]
    weak_refs = ctrl_block["weak_refs_"]["_M_i"]
    user_refs = ctrl_block["user_refs_"]["_M_i"]

    return f"strong={strong_refs} weak={weak_refs} user={user_refs}"


class SharedPtrPrinter(gdb.ValuePrinter):
    """Printer class for a legate::SharedPtr and legate::InternalSharedPtr."""

    def __init__(self, val: Value):
        self._orig_val = val
        self._val = val

        # if val.type is a reference, then val.type.tag is None
        # so we reduce to basic type which removes the reference
        # to get valid string representation of type
        if SHARED_PTR_REGEX.match(gdb.types.get_basic_type(val.type).tag):
            self._val = self._val["ptr_"]

    def children(self) -> Iterator[tuple[str, Value]]:
        """Return an iterator providing the pointer address and value."""
        ptr = self._val["ptr_"].cast(gdb.lookup_type("void").pointer())
        value = self._val["ptr_"].dereference()
        return iter([("ptr_", ptr), ("*ptr_", value)])

    def to_string(self) -> str:
        """Return header string representation of the internal shared ptr."""
        type_str = get_type_str(self._orig_val)
        ptr_str = internal_ptr_string(self._val)
        return f"{type_str} {ptr_str}"

    def display_hint(self) -> str:
        """Return display hint for the internal shared ptr."""
        return "string"


def register_printer(pp: RegexpCollectionPrettyPrinter) -> None:
    """Register legate::InternalSharedPtr and legate::SharedPtr printers."""
    pp.add_printer(
        "SharedPtr", "^legate::(Internal)?SharedPtr<.*>$", SharedPtrPrinter
    )
