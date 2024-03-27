# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from ..type.type_info import Type
    from .logical_store import LogicalStore
    from .shape import Shape

class LogicalArray:
    @staticmethod
    def from_store(store: LogicalStore) -> LogicalArray: ...
    @staticmethod
    def from_raw_handle(raw_handle: int) -> LogicalArray: ...
    @property
    def shape(self) -> Shape: ...
    @property
    def ndim(self) -> int: ...
    @property
    def type(self) -> Type: ...
    @property
    def extents(self) -> tuple[int, ...]: ...
    @property
    def volume(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def unbound(self) -> bool: ...
    @property
    def nullable(self) -> bool: ...
    @property
    def nested(self) -> bool: ...
    @property
    def num_children(self) -> int: ...
    def promote(self, extra_dim: int, dim_size: int) -> LogicalArray: ...
    def project(self, dim: int, index: int) -> LogicalArray: ...
    def slice(self, dim: int, sl: slice) -> LogicalArray: ...
    def transpose(self, axes: Iterable[int]) -> LogicalArray: ...
    def delinearize(self, dim: int, shape: Iterable[int]) -> LogicalArray: ...
    @property
    def data(self) -> LogicalStore: ...
    @property
    def null_mask(self) -> LogicalStore: ...
    def child(self, index: int) -> LogicalArray: ...
    @property
    def raw_handle(self) -> int: ...
