# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lldb import SBValue

INPLACE_VECTOR_REGEX = re.compile(r"cuda.*inplace_vector<.*>")


def unpack_variant_darwin(valobj: SBValue) -> SBValue:
    """Unpacks a variant object into underlying type for Darwin STL."""
    valobj = valobj.GetChildMemberWithName("__impl_")
    variant_index = valobj.GetChildMemberWithName(
        "__index"
    ).GetValueAsUnsigned(0)
    storage_type = valobj.GetType().GetTemplateArgumentType(variant_index)
    return valobj.GetChildAtIndex(0).Cast(storage_type).Clone("Value")


def unpack_variant_llvm(valobj: SBValue) -> SBValue:
    """Unpacks a variant object into underlying type for LLVM STL."""
    variant_index = valobj.GetChildMemberWithName(
        "_M_index"
    ).GetValueAsUnsigned(0)
    storage_type = valobj.GetType().GetTemplateArgumentType(variant_index)
    return valobj.GetChildAtIndex(0).Cast(storage_type).Clone("Value")


if sys.platform == "darwin":
    unpack_variant = unpack_variant_darwin
else:
    unpack_variant = unpack_variant_llvm


class SmallVectorChildrenProvider:
    def __init__(self, valobj: SBValue, internal_dict: dict[str, Any]):  # noqa: ARG002
        """Initialize the children provider for legate::detail::SmallVector."""
        self._valobj = valobj
        self.update()

    def update(self) -> None:
        """Update underlying vector implementation details for printing."""
        # get underlying vector implementation
        variant_val = self._valobj.GetChildMemberWithName("data_")
        self._inplace_vec = unpack_variant(variant_val)

        vec_type = self._inplace_vec.GetType().GetName()
        self._is_small = INPLACE_VECTOR_REGEX.match(vec_type) is not None

        val_type = self._valobj.GetType()
        if val_type.IsReferenceType():
            val_type = val_type.GetPointeeType()
        self._elem_type = val_type.GetTemplateArgumentType(0)

    def num_children(self) -> int:
        """Return the number of children for a legate::detail::SmallVector."""
        if self._is_small:
            return self._inplace_vec.GetChildMemberWithName(
                "__size_"
            ).GetValueAsUnsigned(0)
        return self._inplace_vec.GetNumChildren()

    def get_child_index(self, name: str) -> int:
        """Return the index of a child for a legate::detail::SmallVector."""
        if self._is_small:
            if name.isdigit():
                return int(name)
            error_msg = f"Expected numerical child name, got {name}"
            raise ValueError(error_msg)
        return self._inplace_vec.GetIndexOfChildWithName(name)

    def get_child_at_index(self, index: int) -> SBValue:
        """Return the child at index for a legate::detail::SmallVector."""
        if self._is_small:
            elems_val = self._inplace_vec.GetChildMemberWithName("__elems_")
            elems_val = elems_val.Cast(
                self._elem_type.GetArrayType(self.num_children())
            )
            return elems_val.GetChildAtIndex(index)
        return self._inplace_vec.GetChildAtIndex(index)


def smallvector_summary_formatter(
    valobj: SBValue,
    internal_dict: dict[str, Any],  # noqa: ARG001
) -> str:
    """Return legate::detail::SmallVector summary with size and mode."""
    variant_val = valobj.GetNonSyntheticValue().GetChildMemberWithName("data_")
    vec = unpack_variant(variant_val)
    vec_type = vec.GetType().GetName()
    mode_str = "small" if INPLACE_VECTOR_REGEX.match(vec_type) else "BIG"

    size = valobj.GetNumChildren()
    return f"size={size} mode={mode_str}"
