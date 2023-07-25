# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import FieldSpace


class FieldID:
    def __init__(self, field_space: FieldSpace, fid: int, type: Any) -> None:
        """
        A FieldID class wraps a `legion_field_id_t` in the Legion C API.
        It provides a canonical way to represent an allocated field in a
        field space and means by which to deallocate the field.

        Parameters
        ----------
        field_space : FieldSpace
            The owner field space for this field
        fid : int
            The ID for this field
        type : type
            The type of this field
        """
        self.field_space = field_space
        self._type = type
        self.field_id = fid

    def destroy(self, unordered: bool = False) -> None:
        """
        Deallocate this field from the field space
        """
        self.field_space.destroy_field(self.field_id, unordered)

    @property
    def fid(self) -> int:
        return self.field_id

    @property
    def type(self) -> Any:
        return self._type
