# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lldb import SBValue

SHARED_PTR_REGEX = re.compile(r"^legate::SharedPtr<.*>$")


def extract_atomic_uint(valobj: SBValue) -> int:
    """Extract the atomic integer value from a std::atomic<unsigned int>."""
    return valobj.GetChildAtIndex(0).GetChildAtIndex(0).GetValueAsUnsigned(0)


def _internal_ptr_string(valobj: SBValue) -> str:
    """Return a string representation of legate::InternalSharedPtr."""
    ctrl_block = valobj.GetChildMemberWithName("ctrl_")

    strong_refs = extract_atomic_uint(
        ctrl_block.GetChildMemberWithName("strong_refs_")
    )
    weak_refs = extract_atomic_uint(
        ctrl_block.GetChildMemberWithName("weak_refs_")
    )
    user_refs = extract_atomic_uint(
        ctrl_block.GetChildMemberWithName("user_refs_")
    )

    return f"strong={strong_refs} weak={weak_refs} user={user_refs}"


class SharedPtrChildrenProvider:
    def __init__(self, valobj: SBValue, internal_dict: dict[str, Any]):  # noqa: ARG002
        """Init provider for legate::SharedPtr or legate::InternalSharedPtr.

        Provider provides the pointer and value as its two children.
        """
        self._valobj = valobj
        self.update()

    def update(self) -> None:
        """Update provider for underlying value."""
        if SHARED_PTR_REGEX.match(self._valobj.GetType().GetName()):
            self._valobj = self._valobj.GetChildAtIndex(0)
        self.ptr = self._valobj.GetChildMemberWithName("ptr_")

    def num_children(self) -> int:
        """Return number of children which are the pointer and value."""
        return 2

    def get_child_index(self, name: str) -> int:
        """Return the index of a child."""
        if name == "ptr_":
            return 0
        if name == "value":
            return 1
        error_msg = f"Expected 'ptr_' or 'value', got {name}"
        raise ValueError(error_msg)

    def get_child_at_index(self, index: int) -> SBValue:
        """Return the child at a given index."""
        if index == 0:
            return self.ptr
        if index == 1:
            return self.ptr.Dereference()
        error_msg = f"Expected 0 or 1, got {index}"
        raise ValueError(error_msg)


def sharedptr_summary_formatter(
    valobj: SBValue,
    internal_dict: dict[str, Any],  # noqa: ARG001
) -> str:
    """Return legate::SharedPtr or legate::InternalSharedPtr summary.

    Summary contains weak, strong, and user ref counts.
    """
    valobj = valobj.GetNonSyntheticValue()
    if SHARED_PTR_REGEX.match(valobj.GetType().GetName()):
        valobj = valobj.GetChildAtIndex(0)
    return _internal_ptr_string(valobj)
