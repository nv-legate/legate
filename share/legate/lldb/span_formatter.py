# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lldb import SBValue


class SpanChildrenProvider:
    def __init__(self, valobj: SBValue, internal_dict: dict[str, Any]):  # noqa: ARG002
        """Initialize the children provider for legate::Span."""
        self._valobj = valobj
        self.update()

    def update(self) -> None:
        """Update underlying span implementation details for printing."""
        self._start = self._valobj.GetChildMemberWithName("__data_")
        self._size = self._valobj.GetChildMemberWithName(
            "__size_"
        ).GetValueAsUnsigned(0)

    def num_children(self) -> int:
        """Return the number of children for a legate::Span."""
        return self._size

    def get_child_index(self, name: str) -> int:
        """Return the index of a child for a legate::Span."""
        if name.isdigit():
            return int(name)
        error_msg = f"Expected numerical child name, got {name}"
        raise ValueError(error_msg)

    def get_child_at_index(self, index: int) -> SBValue:
        """Return the child at index for a legate::Span."""
        child_type = self._start.GetType().GetPointeeType()
        offset = index * child_type.GetByteSize()
        return self._start.CreateChildAtOffset(
            f"[{index}]", offset, child_type
        )


def span_summary_formatter(
    valobj: SBValue,
    internal_dict: dict[str, Any],  # noqa: ARG001
) -> str:
    """Return legate::Span summary with size and mode."""
    return f"size={valobj.GetNumChildren()}"
