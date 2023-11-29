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

from typing import Any, Callable, Iterable, Optional, Union

from ...shape import Shape
from ...utils import AnyCallable, ShutdownCallback
from ..data.logical_array import LogicalArray
from ..data.logical_store import LogicalStore
from ..data.scalar import Scalar
from ..mapping.machine import Machine
from ..operation.task import AutoTask, ManualTask
from ..type.type_info import Type
from .library import Library

class Runtime:
    def find_library(self, library_name: str) -> Library: ...
    @property
    def core_library(self) -> Library: ...
    def create_auto_task(self, library: Library, task_id: int) -> AutoTask: ...
    def create_manual_task(
        self,
        library: Library,
        task_id: int,
        launch_shape: Iterable[int],
        lower_bounds: Optional[Iterable[int]] = None,
    ) -> ManualTask: ...
    def issue_copy(
        self,
        target: LogicalStore,
        source: LogicalStore,
        redop: Optional[int] = None,
    ) -> None: ...
    def issue_gather(
        self,
        target: LogicalStore,
        source: LogicalStore,
        source_indirect: LogicalStore,
        redop: Optional[int] = None,
    ) -> None: ...
    def issue_scatter(
        self,
        target: LogicalStore,
        target_indirect: LogicalStore,
        source: LogicalStore,
        redop: Optional[int] = None,
    ) -> None: ...
    def issue_scatter_gather(
        self,
        target: LogicalStore,
        target_indirect: LogicalStore,
        source: LogicalStore,
        source_indirect: LogicalStore,
        redop: Optional[int] = None,
    ) -> None: ...
    def issue_fill(
        self, lhs: LogicalStore, value: Union[LogicalStore, Scalar]
    ) -> None: ...
    def tree_reduce(
        self,
        library: Library,
        task_id: int,
        store: LogicalStore,
        radix: int = 4,
    ) -> LogicalStore: ...
    def submit(self, task: Union[AutoTask, ManualTask]) -> None: ...
    def create_array(
        self,
        dtype: Type,
        shape: Optional[Union[Shape, Iterable[int]]] = None,
        nullable: bool = False,
        optimize_scalar: bool = False,
        ndim: Optional[int] = None,
    ) -> LogicalArray: ...
    def create_array_like(
        self, array: LogicalArray, dtype: Type
    ) -> LogicalArray: ...
    def create_store(
        self,
        dtype: Type,
        shape: Optional[Union[Shape, Iterable[int]]] = None,
        optimize_scalar: bool = False,
        ndim: Optional[int] = None,
    ) -> LogicalStore: ...
    def create_store_from_scalar(
        self,
        scalar: Scalar,
        shape: Optional[Union[Shape, Iterable[int]]] = None,
    ) -> LogicalStore: ...
    def create_store_from_bytes(
        self,
        dtype: Type,
        shape: Optional[Union[Shape, Iterable[int]]],
        data: bytes,
        share: bool,
    ) -> LogicalStore: ...
    def create_store_from_buffer(
        self,
        dtype: Type,
        shape: Optional[Union[Shape, Iterable[int]]],
        data: object,
        share: bool,
    ) -> LogicalStore: ...
    @property
    def max_pending_exceptions(self) -> int: ...
    def raise_pending_task_exception(self) -> None: ...
    def issue_execution_fence(self, block: bool = False) -> None: ...
    def get_machine(self) -> Machine: ...
    @property
    def machine(self) -> Machine: ...
    def add_shutdown_callback(self, callback: ShutdownCallback) -> None: ...

def get_legate_runtime() -> Runtime: ...
def get_machine() -> Machine: ...
def track_provenance(
    nested: bool = False,
) -> Callable[[AnyCallable], AnyCallable]: ...
