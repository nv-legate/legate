# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Union

import legate.core.types as ty

from . import Future, FutureMap, Point, Rect
from .constraints import PartSym
from .launcher import CopyLauncher, TaskLauncher
from .partition import REPLICATE, Weighted
from .shape import Shape
from .store import Store, StorePartition
from .utils import OrderedSet

if TYPE_CHECKING:
    from .communicator import Communicator
    from .constraints import Constraint
    from .context import Context
    from .solver import Strategy
    from .types import DTType

ProjFunc = Callable[[Point], Point]


class Operation:
    def __init__(
        self, context: Context, mapper_id: int = 0, op_id: int = 0
    ) -> None:
        self._context = context
        self._mapper_id = mapper_id
        self._op_id = op_id
        self._inputs: list[Store] = []
        self._outputs: list[Store] = []
        self._reductions: list[tuple[Store, int]] = []
        self._input_parts: list[PartSym] = []
        self._output_parts: list[PartSym] = []
        self._reduction_parts: list[PartSym] = []
        self._unbound_outputs: list[int] = []
        self._scalar_outputs: list[int] = []
        self._scalar_reductions: list[int] = []
        self._partitions: dict[Store, list[PartSym]] = {}
        self._constraints: list[Constraint] = []
        self._all_parts: list[PartSym] = []
        self._launch_domain: Union[Rect, None] = None
        self._error_on_interference = True

    @property
    def context(self) -> Context:
        return self._context

    @property
    def mapper_id(self) -> int:
        return self._mapper_id

    @property
    def inputs(self) -> list[Store]:
        return self._inputs

    @property
    def outputs(self) -> list[Store]:
        return self._outputs

    @property
    def reductions(self) -> list[tuple[Store, int]]:
        return self._reductions

    @property
    def unbound_outputs(self) -> list[int]:
        return self._unbound_outputs

    @property
    def scalar_outputs(self) -> list[int]:
        return self._scalar_outputs

    @property
    def scalar_reductions(self) -> list[int]:
        return self._scalar_reductions

    @property
    def constraints(self) -> list[Constraint]:
        return self._constraints

    @property
    def all_unknowns(self) -> list[PartSym]:
        return self._all_parts

    def get_all_stores(self) -> OrderedSet[Store]:
        result: OrderedSet[Store] = OrderedSet()
        result.update(self._inputs)
        result.update(self._outputs)
        result.update(store for (store, _) in self._reductions)
        return result

    @staticmethod
    def _check_store(store: Store, allow_unbound: bool = False) -> None:
        if not isinstance(store, Store):
            raise ValueError(f"Expected a Store, but got {type(store)}")
        elif not allow_unbound and store.unbound:
            raise ValueError("Expected a bound Store")

    def _get_unique_partition(self, store: Store) -> PartSym:
        if store not in self._partitions:
            return self.declare_partition(store)

        parts = self._partitions[store]
        if len(parts) > 1:
            raise RuntimeError(
                "Ambiguous store argument. When multple partitions exist for "
                "this store, a partition should be specified."
            )
        return parts[0]

    def add_input(
        self, store: Store, partition: Optional[PartSym] = None
    ) -> None:
        self._check_store(store)
        if partition is None:
            partition = self._get_unique_partition(store)
        self._inputs.append(store)
        self._input_parts.append(partition)

    def add_output(
        self, store: Store, partition: Optional[PartSym] = None
    ) -> None:
        self._check_store(store, allow_unbound=True)
        if store.kind is Future:
            self._scalar_outputs.append(len(self._outputs))
        elif store.unbound:
            self._unbound_outputs.append(len(self._outputs))
        if partition is None:
            partition = self._get_unique_partition(store)
        self._outputs.append(store)
        self._output_parts.append(partition)

    def add_reduction(
        self, store: Store, redop: int, partition: Optional[PartSym] = None
    ) -> None:
        self._check_store(store)
        if store.kind is Future:
            self._scalar_reductions.append(len(self._reductions))
        if partition is None:
            partition = self._get_unique_partition(store)
        self._reductions.append((store, redop))
        self._reduction_parts.append(partition)

    def add_alignment(self, store1: Store, store2: Store) -> None:
        self._check_store(store1)
        self._check_store(store2)
        if store1.shape != store2.shape:
            raise ValueError(
                "Stores must have the same shape to be aligned, "
                f"but got {store1.shape} and {store2.shape}"
            )
        part1 = self._get_unique_partition(store1)
        part2 = self._get_unique_partition(store2)
        self.add_constraint(part1 == part2)

    def add_broadcast(
        self, store: Store, axes: Optional[Union[int, Iterable[int]]] = None
    ) -> None:
        self._check_store(store)
        part = self._get_unique_partition(store)
        self.add_constraint(part.broadcast(axes=axes))

    def add_constraint(self, constraint: Constraint) -> None:
        self._constraints.append(constraint)

    def execute(self) -> None:
        self._context.runtime.submit(self)

    def get_tag(self, strategy: Strategy, part: Any) -> int:
        if strategy.is_key_part(part):
            return 1  # LEGATE_CORE_KEY_STORE_TAG
        else:
            return 0

    def _get_symbol_id(self) -> int:
        return len(self._all_parts)

    def get_name(self) -> str:
        libname = self.context.library.get_name()
        return f"{libname}.Operation(uid:{self._op_id})"

    def declare_partition(
        self, store: Store, disjoint: bool = True, complete: bool = True
    ) -> PartSym:
        sym = PartSym(
            self._op_id,
            self.get_name(),
            store,
            self._get_symbol_id(),
            disjoint=disjoint,
            complete=complete,
        )
        if store not in self._partitions:
            self._partitions[store] = [sym]
        else:
            self._partitions[store].append(sym)
        self._all_parts.append(sym)
        return sym


class Task(Operation):
    def __init__(
        self,
        context: Context,
        task_id: int,
        mapper_id: int = 0,
        op_id: int = 0,
    ) -> None:
        super().__init__(context, mapper_id=mapper_id, op_id=op_id)
        self._task_id = task_id
        self._scalar_args: list[tuple[Any, DTType]] = []
        self._comm_args: list[Communicator] = []

    @property
    def uses_communicator(self) -> bool:
        return len(self._comm_args) > 0

    def get_name(self) -> str:
        libname = self.context.library.get_name()
        return f"{libname}.Task(tid:{self._task_id}, uid:{self._op_id})"

    def add_scalar_arg(self, value: Any, dtype: DTType) -> None:
        self._scalar_args.append((value, dtype))

    def add_dtype_arg(self, dtype: DTType) -> None:
        code = self._context.type_system[dtype].code
        self._scalar_args.append((code, ty.int32))

    def _add_scalar_args_to_launcher(self, launcher: TaskLauncher) -> None:
        for (arg, dtype) in self._scalar_args:
            launcher.add_scalar_arg(arg, dtype)

    def _demux_scalar_stores(
        self,
        result: Union[Future, FutureMap],
        launch_domain: Union[Rect, None],
    ) -> None:
        num_unbound_outs = len(self.unbound_outputs)
        num_scalar_outs = len(self.scalar_outputs)
        num_scalar_reds = len(self.scalar_reductions)
        runtime = self.context.runtime

        num_all_scalars = num_unbound_outs + num_scalar_outs + num_scalar_reds
        launch_shape = (
            None
            if launch_domain is None
            else Shape(c + 1 for c in launch_domain.hi)
        )

        if num_all_scalars == 0:
            return
        elif num_all_scalars == 1:
            if num_scalar_outs == 1:
                output = self.outputs[self.scalar_outputs[0]]
                output.set_storage(result)
            elif num_scalar_reds == 1:
                (output, redop) = self.reductions[self.scalar_reductions[0]]
                redop_id = output.type.reduction_op_id(redop)
                output.set_storage(runtime.reduce_future_map(result, redop_id))
            elif launch_shape is not None:
                assert num_unbound_outs == 1
                assert isinstance(result, FutureMap)
                output = self.outputs[self.unbound_outputs[0]]
                # TODO: need to track partitions for N-D unbound stores
                if output.ndim == 1:
                    partition = Weighted(runtime, launch_shape, result)
                    output.set_key_partition(partition)
        else:
            idx = 0
            # TODO: We can potentially deduplicate these extraction tasks
            # by grouping output stores that are mapped to the same field space
            if launch_shape is not None:
                for out_idx in self.unbound_outputs:
                    output = self.outputs[out_idx]
                    # TODO: need to track partitions for N-D unbound stores
                    if output.ndim > 1:
                        continue
                    weights = runtime.extract_scalar_with_domain(
                        result, idx, launch_domain
                    )
                    partition = Weighted(runtime, launch_shape, weights)
                    output.set_key_partition(partition)
                    idx += 1
            else:
                idx += len(self.unbound_outputs)
            for out_idx in self.scalar_outputs:
                output = self.outputs[out_idx]
                data = (
                    runtime.extract_scalar(result, idx)
                    if launch_domain is None
                    else runtime.extract_scalar_with_domain(
                        result, idx, launch_domain
                    )
                )
                output.set_storage(data)
                idx += 1
            for red_idx in self.scalar_reductions:
                (output, redop) = self.reductions[red_idx]
                redop_id = output.type.reduction_op_id(redop)
                data = (
                    runtime.extract_scalar(result, idx)
                    if launch_domain is None
                    else runtime.extract_scalar_with_domain(
                        result, idx, launch_domain
                    )
                )
                output.set_storage(runtime.reduce_future_map(data, redop_id))
                idx += 1

    def add_nccl_communicator(self) -> None:
        comm = self._context.get_nccl_communicator()
        self._comm_args.append(comm)

    def _add_communicators(
        self, launcher: TaskLauncher, launch_domain: Union[Rect, None]
    ) -> None:
        if launch_domain is None:
            return
        for comm in self._comm_args:
            handle = comm.get_handle(launch_domain)
            launcher.add_communicator(handle)


class AutoTask(Task):
    def __init__(
        self,
        context: Context,
        task_id: int,
        mapper_id: int = 0,
        op_id: int = 0,
    ) -> None:
        super().__init__(context, task_id, mapper_id, op_id)

    def launch(self, strategy: Strategy) -> None:
        launcher = TaskLauncher(self.context, self._task_id, self.mapper_id)

        def get_requirement(
            store: Store, part_symb: PartSym
        ) -> tuple[Any, int, StorePartition]:
            store_part = store.partition(strategy.get_partition(part_symb))
            req = store_part.get_requirement(strategy.launch_ndim)
            tag = self.get_tag(strategy, part_symb)
            return req, tag, store_part

        for store, part_symb in zip(self._inputs, self._input_parts):
            req, tag, _ = get_requirement(store, part_symb)
            launcher.add_input(store, req, tag=tag)

        for store, part_symb in zip(self._outputs, self._output_parts):
            if store.unbound:
                continue
            req, tag, store_part = get_requirement(store, part_symb)
            launcher.add_output(store, req, tag=tag)
            # We update the key partition of a store only when it gets updated
            store.set_key_partition(store_part.partition)

        for ((store, redop), part_symb) in zip(
            self._reductions, self._reduction_parts
        ):
            req, tag, store_part = get_requirement(store, part_symb)

            can_read_write = store_part.is_disjoint_for(strategy.launch_domain)
            req.redop = store.type.reduction_op_id(redop)

            launcher.add_reduction(
                store, req, tag=tag, read_write=can_read_write
            )

        for (store, part_symb) in zip(self._outputs, self._output_parts):
            if not store.unbound:
                continue
            fspace = strategy.get_field_space(part_symb)
            field_id = fspace.allocate_field(store.type)
            launcher.add_unbound_output(store, fspace, field_id)

        self._add_scalar_args_to_launcher(launcher)

        launch_domain = strategy.launch_domain if strategy.parallel else None
        self._add_communicators(launcher, launch_domain)

        # TODO: For now we make sure no other operations are interleaved with
        # the set of tasks that use a communicator. In the future, the
        # communicator monad will do this for us.
        if self.uses_communicator:
            self._context.issue_execution_fence()

        result: Union[Future, FutureMap]
        if launch_domain is not None:
            result = launcher.execute(launch_domain)
        else:
            result = launcher.execute_single()

        if self.uses_communicator:
            self._context.issue_execution_fence()

        self._demux_scalar_stores(result, launch_domain)


class ManualTask(Task):
    def __init__(
        self,
        context: Context,
        task_id: int,
        launch_domain: Rect,
        mapper_id: int = 0,
        op_id: int = 0,
    ) -> None:
        super().__init__(context, task_id, mapper_id, op_id)
        self._launch_domain: Rect = launch_domain
        self._input_projs: list[Union[ProjFunc, None]] = []
        self._output_projs: list[Union[ProjFunc, None]] = []
        self._reduction_projs: list[Union[ProjFunc, None]] = []

        self._input_parts: list[StorePartition] = []  # type: ignore [assignment] # noqa: E501
        self._output_parts: list[StorePartition] = []  # type: ignore [assignment] # noqa: E501
        self._reduction_parts: list[tuple[StorePartition, int]] = []  # type: ignore [assignment] # noqa: E501

    @property
    def launch_ndim(self) -> int:
        return self._launch_domain.dim

    def get_all_stores(self) -> OrderedSet[Store]:
        return OrderedSet()

    @staticmethod
    def _check_arg(arg: Union[Store, StorePartition]) -> None:
        if not isinstance(arg, (Store, StorePartition)):
            raise ValueError(
                f"Expected a Store or StorePartition, but got {type(arg)}"
            )

    def add_input(  # type: ignore [override]
        self,
        arg: Union[Store, StorePartition],
        proj: Optional[ProjFunc] = None,
    ) -> None:
        self._check_arg(arg)
        if isinstance(arg, Store):
            self._input_parts.append(arg.partition(REPLICATE))
        else:
            self._input_parts.append(arg)
        self._input_projs.append(proj)

    def add_output(  # type: ignore [override]
        self,
        arg: Union[Store, StorePartition],
        proj: Optional[ProjFunc] = None,
    ) -> None:
        self._check_arg(arg)
        if isinstance(arg, Store):
            if arg.unbound:
                raise ValueError(
                    "Unbound store cannot be used with "
                    "manually parallelized task"
                )
            if arg.kind is Future:
                self._scalar_outputs.append(len(self._outputs))
            self._output_parts.append(arg.partition(REPLICATE))
        else:
            self._output_parts.append(arg)
        self._output_projs.append(proj)

    def add_reduction(  # type: ignore [override]
        self,
        arg: Union[Store, StorePartition],
        redop: int,
        proj: Optional[ProjFunc] = None,
    ) -> None:
        self._check_arg(arg)
        if isinstance(arg, Store):
            if arg.kind is Future:
                self._scalar_reductions.append(len(self._reductions))
            self._reduction_parts.append((arg.partition(REPLICATE), redop))
        else:
            self._reduction_parts.append((arg, redop))
        self._reduction_projs.append(proj)

    def add_alignment(self, store1: Store, store2: Store) -> None:
        raise TypeError(
            "Partitioning constraints are not allowed for "
            "manually parallelized tasks"
        )

    def add_broadcast(
        self, store: Store, axes: Optional[Union[int, Iterable[int]]] = None
    ) -> None:
        raise TypeError(
            "Partitioning constraints are not allowed for "
            "manually parallelized tasks"
        )

    def add_constraint(self, constraint: Constraint) -> None:
        raise TypeError(
            "Partitioning constraints are not allowed for "
            "manually parallelized tasks"
        )

    def launch(self, strategy: Strategy) -> None:
        tag = self.context.core_library.LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG
        launcher = TaskLauncher(
            self.context,
            self._task_id,
            self.mapper_id,
            error_on_interference=False,
            tag=tag,
        )

        for part, proj_fn in zip(self._input_parts, self._input_projs):
            req = part.get_requirement(self.launch_ndim, proj_fn)
            launcher.add_input(part.store, req, tag=0)

        for part, proj_fn in zip(self._output_parts, self._output_projs):
            req = part.get_requirement(self.launch_ndim, proj_fn)
            launcher.add_output(part.store, req, tag=0)

        for (part, redop), proj_fn in zip(
            self._reduction_parts, self._reduction_projs
        ):
            req = part.get_requirement(self.launch_ndim, proj_fn)
            req.redop = part.store.type.reduction_op_id(redop)
            can_read_write = part.is_disjoint_for(self._launch_domain)
            launcher.add_reduction(
                part.store, req, tag=0, read_write=can_read_write
            )

        self._add_scalar_args_to_launcher(launcher)

        self._add_communicators(launcher, self._launch_domain)

        # TODO: For now we make sure no other operations are interleaved with
        # the set of tasks that use a communicator. In the future, the
        # communicator monad will do this for us.
        if self.uses_communicator:
            self._context.issue_execution_fence()

        result = launcher.execute(self._launch_domain)

        if self.uses_communicator:
            self._context.issue_execution_fence()

        self._demux_scalar_stores(result, self._launch_domain)


class Copy(Operation):
    def __init__(self, context: Context, mapper_id: int = 0) -> None:
        super().__init__(context, mapper_id)
        self._source_indirects: list[Store] = []
        self._target_indirects: list[Store] = []
        self._source_indirect_parts: list[PartSym] = []
        self._target_indirect_parts: list[PartSym] = []

    def get_name(self) -> str:
        libname = self.context.library.get_name()
        return f"{libname}.Copy(uid:{self._op_id})"

    @property
    def inputs(self) -> list[Store]:
        return super().inputs + self._source_indirects + self._target_indirects

    def add_output(
        self, store: Store, partition: Optional[PartSym] = None
    ) -> None:
        if len(self._reductions) > 0:
            raise RuntimeError(
                "Copy targets must be either all normal outputs or reductions"
            )
        super().add_output(store, partition)

    def add_reduction(
        self, store: Store, redop: int, partition: Optional[PartSym] = None
    ) -> None:
        if len(self._outputs) > 0:
            raise RuntimeError(
                "Copy targets must be either all normal outputs or reductions"
            )
        super().add_reduction(store, redop, partition)

    def add_source_indirect(
        self, store: Store, partition: Optional[PartSym] = None
    ) -> None:
        self._check_store(store)
        if partition is None:
            partition = self._get_unique_partition(store)
        self._source_indirects.append(store)
        self._source_indirect_parts.append(partition)

    def add_target_indirect(
        self, store: Store, partition: Optional[PartSym] = None
    ) -> None:
        self._check_store(store)
        if partition is None:
            partition = self._get_unique_partition(store)
        self._target_indirects.append(store)
        self._target_indirect_parts.append(partition)

    @property
    def constraints(self) -> list[Constraint]:
        constraints: list[Constraint] = []
        if len(self._source_indirects) + len(self._target_indirects) == 0:
            for src, tgt in zip(self._input_parts, self._output_parts):
                if src.store.shape != tgt.store.shape:
                    raise ValueError(
                        "Each output must have the same shape as the "
                        f"input, but got {tuple(src.store.shape)} and "
                        f"{tuple(tgt.store.shape)}"
                    )
                constraints.append(src == tgt)
        else:
            if len(self._source_indirects) > 0:
                output_parts = (
                    self._output_parts
                    if len(self._outputs) > 0
                    else self._reduction_parts
                )
                for src, tgt in zip(self._source_indirect_parts, output_parts):
                    if src.store.shape != tgt.store.shape:
                        raise ValueError(
                            "Each output must have the same shape as the "
                            "corresponding source indirect field, but got "
                            f"{tuple(src.store.shape)} and "
                            f"{tuple(tgt.store.shape)}"
                        )
                    constraints.append(src == tgt)
            if len(self._target_indirects) > 0:
                for src, tgt in zip(
                    self._input_parts, self._target_indirect_parts
                ):
                    if src.store.shape != tgt.store.shape:
                        raise ValueError(
                            "Each input must have the same shape as the "
                            "corresponding target indirect field, but got "
                            f"{tuple(src.store.shape)} and "
                            f"{tuple(tgt.store.shape)}"
                        )
                    constraints.append(src == tgt)
        return constraints

    def add_alignment(self, store1: Store, store2: Store) -> None:
        raise TypeError(
            "User partitioning constraints are not allowed for copies"
        )

    def add_broadcast(
        self, store: Store, axes: Optional[Union[int, Iterable[int]]] = None
    ) -> None:
        raise TypeError(
            "User partitioning constraints are not allowed for copies"
        )

    def add_constraint(self, constraint: Constraint) -> None:
        raise TypeError(
            "User partitioning constraints are not allowed for copies"
        )

    def launch(self, strategy: Strategy) -> None:
        launcher = CopyLauncher(self.context, self.mapper_id)

        assert len(self._inputs) == len(self._outputs) or len(
            self._inputs
        ) == len(self._reductions)
        assert len(self._source_indirects) == 0 or len(
            self._source_indirects
        ) == len(self._inputs)
        assert len(self._target_indirects) == 0 or len(
            self._target_indirects
        ) == len(self._outputs)

        def get_requirement(
            store: Store, part_symb: PartSym
        ) -> tuple[Any, int, StorePartition]:
            store_part = store.partition(strategy.get_partition(part_symb))
            req = store_part.get_requirement(strategy.launch_ndim)
            tag = self.get_tag(strategy, part_symb)
            return req, tag, store_part

        for store, part_symb in zip(self._inputs, self._input_parts):
            req, tag, _ = get_requirement(store, part_symb)
            launcher.add_input(store, req, tag=tag)

        for store, part_symb in zip(self._outputs, self._output_parts):
            assert not store.unbound
            req, tag, store_part = get_requirement(store, part_symb)
            launcher.add_output(store, req, tag=tag)

        for ((store, redop), part_symb) in zip(
            self._reductions, self._reduction_parts
        ):
            req, tag, store_part = get_requirement(store, part_symb)
            req.redop = store.type.reduction_op_id(redop)
            launcher.add_reduction(store, req, tag=tag)
        for store, part_symb in zip(
            self._source_indirects, self._source_indirect_parts
        ):
            req, tag, store_part = get_requirement(store, part_symb)
            launcher.add_source_indirect(store, req, tag=tag)
        for store, part_symb in zip(
            self._target_indirects, self._target_indirect_parts
        ):
            req, tag, store_part = get_requirement(store, part_symb)
            launcher.add_target_indirect(store, req, tag=tag)

        launch_domain = strategy.launch_domain if strategy.parallel else None
        if launch_domain is not None:
            launcher.execute(launch_domain)
        else:
            launcher.execute_single()


class _RadixProj:
    def __init__(self, radix: int, offset: int) -> None:
        self._radix = radix
        self._offset = offset

    def __call__(self, p: tuple[int, ...]) -> tuple[int]:
        return (p[0] * self._radix + self._offset,)


class Reduce(Task):
    def __init__(
        self,
        context: Context,
        task_id: int,
        radix: int,
        mapper_id: int,
        op_id: int,
    ) -> None:
        super().__init__(context, task_id, mapper_id, op_id)
        self._runtime = context.runtime
        self._radix = radix

    def launch(self, strategy: Strategy) -> None:
        assert len(self._inputs) == 1 and len(self._outputs) == 1

        result = self._outputs[0]

        output = self._inputs[0]
        opart = output.partition(strategy.get_partition(self._input_parts[0]))

        done = False
        launch_domain = strategy.launch_domain if strategy.parallel else None
        fan_in = (
            strategy.launch_domain.get_volume() if strategy.parallel else 1
        )

        proj_fns = list(
            _RadixProj(self._radix, off) for off in range(self._radix)
        )

        while not done:
            input = output
            ipart = opart

            tag = self.context.core_library.LEGATE_CORE_TREE_REDUCE_TAG
            launcher = TaskLauncher(
                self.context, self._task_id, self.mapper_id, tag=tag
            )

            for proj_fn in proj_fns:
                launcher.add_input(input, ipart.get_requirement(1, proj_fn))

            output = self._context.create_store(input.type)
            fspace = self._runtime.create_field_space()
            field_id = fspace.allocate_field(input.type)
            launcher.add_unbound_output(output, fspace, field_id)

            num_tasks = (fan_in + self._radix - 1) // self._radix
            launch_domain = Rect([num_tasks])
            weights = launcher.execute(launch_domain)

            launch_shape = Shape(c + 1 for c in launch_domain.hi)
            weighted = Weighted(self._runtime, launch_shape, weights)
            output.set_key_partition(weighted)
            opart = output.partition(weighted)

            fan_in = num_tasks
            done = fan_in == 1

        result.set_storage(output.storage)
