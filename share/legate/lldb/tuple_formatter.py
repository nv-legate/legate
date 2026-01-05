# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lldb import SBValue


class TupleChildrenProvider:
    def __init__(self, valobj: SBValue, internal_dict: dict[str, Any]):  # noqa: ARG002
        """Initialize the children provider for legate::tuple."""
        self._valobj = valobj
        self.update()

    def update(self) -> None:
        """Update underlying tuple implementation details for printing."""
        # get underlying vector implementation
        vector = self._valobj.GetChildMemberWithName("data_")
        std_vector_type = vector.GetType().GetTypedefedType()
        self._vector = vector.Cast(std_vector_type).Clone("Value")

    def num_children(self) -> int:
        """Return the number of children for a legate::tuple."""
        return self._vector.GetNumChildren()

    def get_child_index(self, name: str) -> int:
        """Return the index of a child for a legate::tuple."""
        return self._vector.GetIndexOfChildWithName(name)

    def get_child_at_index(self, index: int) -> SBValue:
        """Return the child at index for a legate::tuple."""
        return self._vector.GetChildAtIndex(index)


def tuple_summary_formatter(
    valobj: SBValue,
    internal_dict: dict[str, Any],  # noqa: ARG001
) -> str:
    """Return legate::tuple summary with size."""
    return f"size={valobj.GetNumChildren()}"
