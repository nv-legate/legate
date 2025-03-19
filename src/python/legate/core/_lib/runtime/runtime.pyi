# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from typing import Any

from ...utils import AnyCallable, ShutdownCallback
from ..data.logical_array import LogicalArray
from ..data.logical_store import LogicalStore
from ..data.scalar import Scalar
from ..data.shape import Shape
from ..mapping.machine import Machine
from ..operation.task import AutoTask, ManualTask
from ..task.variant_options import VariantOptions
from ..type.types import Type
from ..utilities.typedefs import LocalTaskID, VariantCode
from ..utilities.unconstructable import Unconstructable
from .library import Library
from .resource import ResourceConfig

class Runtime(Unconstructable):
    def find_library(self, library_name: str) -> Library: ...
    def find_or_create_library(
        self,
        library_name: str,
        *,
        config: ResourceConfig | None = None,
        default_options: dict[VariantCode, VariantOptions] | None = None,
    ) -> tuple[Library, bool]: ...
    def create_library(self, library_name: str) -> Library: ...
    @property
    def core_library(self) -> Library: ...
    # This prototype is a lie, technically (in Cython) it's only LocalTaskID,
    # but we allow int as a type-checking convenience to users
    def create_auto_task(
        self, library: Library, task_id: LocalTaskID | int
    ) -> AutoTask: ...
    # This prototype is a lie, technically (in Cython) it's only LocalTaskID,
    # but we allow int as a type-checking convenience to users
    def create_manual_task(
        self,
        library: Library,
        task_id: LocalTaskID | int,
        launch_shape: Collection[int],
        lower_bounds: Collection[int] | None = None,
    ) -> ManualTask: ...
    def issue_copy(
        self,
        target: LogicalStore,
        source: LogicalStore,
        redop: int | None = None,
    ) -> None: ...
    def issue_gather(
        self,
        target: LogicalStore,
        source: LogicalStore,
        source_indirect: LogicalStore,
        redop: int | None = None,
    ) -> None: ...
    def issue_scatter(
        self,
        target: LogicalStore,
        target_indirect: LogicalStore,
        source: LogicalStore,
        redop: int | None = None,
    ) -> None: ...
    def issue_scatter_gather(
        self,
        target: LogicalStore,
        target_indirect: LogicalStore,
        source: LogicalStore,
        source_indirect: LogicalStore,
        redop: int | None = None,
    ) -> None: ...
    def issue_fill(
        self, lhs: LogicalStore | LogicalArray, value: Any
    ) -> None: ...
    # This prototype is a lie, technically (in Cython) it's only LocalTaskID,
    # but we allow int as a type-checking convenience to users
    def tree_reduce(
        self,
        library: Library,
        task_id: LocalTaskID | int,
        store: LogicalStore,
        radix: int = 4,
    ) -> LogicalStore: ...
    def submit(self, task: AutoTask | ManualTask) -> None: ...
    def create_array(
        self,
        dtype: Type,
        shape: Shape | Collection[int] | None = None,
        nullable: bool = False,
        optimize_scalar: bool = False,
        ndim: int | None = None,
    ) -> LogicalArray: ...
    def create_array_like(
        self, array: LogicalArray, dtype: Type | None = None
    ) -> LogicalArray: ...
    def create_store(
        self,
        dtype: Type,
        shape: Shape | Collection[int] | None = None,
        optimize_scalar: bool = False,
        ndim: int | None = None,
    ) -> LogicalStore: ...
    def create_store_from_scalar(
        self, scalar: Scalar, shape: Shape | Collection[int] | None = None
    ) -> LogicalStore: ...
    def create_store_from_buffer(
        self,
        dtype: Type,
        shape: Shape | Collection[int],
        data: object,
        read_only: bool,
    ) -> LogicalStore: ...
    def prefetch_bloated_instances(
        self,
        store: LogicalStore,
        low_offsets: tuple[int, ...],
        high_offsets: tuple[int, ...],
        initialize: bool = False,
    ) -> None: ...
    def issue_mapping_fence(self) -> None: ...
    def issue_execution_fence(self, block: bool = False) -> None: ...
    @property
    def node_count(self) -> int: ...
    @property
    def node_id(self) -> int: ...
    def get_machine(self) -> Machine: ...
    @property
    def machine(self) -> Machine: ...
    def finish(self) -> None: ...
    def add_shutdown_callback(self, callback: ShutdownCallback) -> None: ...

def get_legate_runtime() -> Runtime: ...
def get_machine() -> Machine: ...
def track_provenance(
    nested: bool = False,
) -> Callable[[AnyCallable], AnyCallable]: ...
def is_running_in_task() -> bool: ...
@contextmanager
def ProfileRange(provenance: str) -> Iterator[None]: ...
