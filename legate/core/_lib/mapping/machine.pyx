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

from libc.stdint cimport uint32_t

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union

from libcpp.map cimport map as std_map
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector as std_vector

from .mapping cimport TaskTarget

from ...utils import is_iterable
from .mapping import TaskTarget as PyTaskTarget


class EmptyMachineError(Exception):
    pass


@dataclass(frozen=True)
class ProcessorSlice:
    target: TaskTarget
    slice: slice


PROC_RANGE_KEY = Union[slice, int]
MACHINE_KEY = Union[PyTaskTarget, slice, int, ProcessorSlice]


cdef class ProcessorRange:
    @staticmethod
    def create(uint32_t low, uint32_t high, uint32_t per_node_count):
        return ProcessorRange(low, high, per_node_count)

    @staticmethod
    def create_empty() -> ProcessorRange:
        return ProcessorRange()

    @staticmethod
    cdef ProcessorRange from_handle(_ProcessorRange handle):
        cdef ProcessorRange result = ProcessorRange.__new__(ProcessorRange)
        result._handle = handle
        return result

    def __cinit__(
        self,
        uint32_t low = 0,
        uint32_t high = 0,
        uint32_t per_node_count = 0,
    ) -> None:
        self._handle = _ProcessorRange(low, high, per_node_count)

    @property
    def low(self):
        """
        Returns the lower bound of the processor range

        Returns
        -------
        int
            Lower bound (inclusive)
        """
        return self._handle.low

    @property
    def high(self):
        """
        Returns the upper bound of the processor range

        Returns
        -------
        int
            Upper bound (exclusive)
        """
        return self._handle.high

    @property
    def per_node_count(self):
        """
        Returns the number of processors per node

        Returns
        -------
        int
            Per-node processor count
        """
        return self._handle.per_node_count

    @property
    def count(self):
        """
        Returns the number of processors in the range

        Returns
        -------
        int
            Processor count
        """
        return self._handle.count()

    def __len__(self) -> uint32_t:
        """
        Returns the number of processors in the range

        Returns
        -------
        int
            Processor count
        """
        return self.count

    @property
    def empty(self):
        """
        Indicates if the processor range is empty

        Returns
        -------
        bool
            ``True`` if the machine is empty, ``False`` otherwise.
        """
        return self._handle.empty()

    def slice(self, slice sl) -> ProcessorRange:
        """
        Slices the processor range by a given ``slice``

        Parameters
        ----------
        sl : slice
            A ``slice`` to slice this processor range by

        Returns
        -------
        ProcessorRange
            Processor range after slicing
        """
        if sl.step is not None and sl.step != 1:
            raise ValueError("The slicing step must be 1")
        start = 0 if sl.start is None else sl.start
        stop = self.high if sl.stop is None else sl.stop
        return ProcessorRange.from_handle(self._handle.slice(start, stop))

    def __getitem__(self, key: PROC_RANGE_KEY) -> ProcessorRange:
        """
        Slices the processor range with a given slicer

        Parameters
        ----------
        key : slice, int
            Key to slice the processor range by. If the ``key`` is an ``int``,
            it is treated like a singleton slice (i.e., ```slice(key, key +
            1)```)

        Returns
        -------
        ProcessorRange
            Processor range after slicing
        """
        if isinstance(key, int):
            return self.slice(slice(key, key + 1))
        elif isinstance(key, slice):
            return self.slice(key)

        raise KeyError(f"Invalid slicing key: {key}")

    def get_node_range(self) -> tuple:
        """
        Returns the range of node IDs for this processor range

        Returns
        -------
        tuple[int, int]
            Half-open interval of node IDs
        """
        result = self._handle.get_node_range()
        return (result.low, result.high)

    def __str__(self) -> str:
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        return str(self)

    def __and__(self, other: ProcessorRange) -> ProcessorRange:
        """
        Computes an intersection with a given processor range

        Parameters
        ----------
        other : ProcessorRange
            A processor range to intersect with

        Returns
        -------
        ProcessorRange
            Intersection result
        """
        return ProcessorRange.from_handle(self._handle & other._handle)

    def __eq__(self, other: ProcessorRange) -> bool:
        return self._handle == other._handle

    def __ne__(self, other: ProcessorRange) -> bool:
        return self._handle != other._handle

    def __lt__(self, other: ProcessorRange) -> bool:
        return self._handle < other._handle


cdef class Machine:
    @staticmethod
    cdef Machine from_handle(_Machine handle):
        cdef Machine result = Machine.__new__(Machine)
        result._handle = handle
        return result

    def __cinit__(self, ranges = None) -> None:
        if ranges is None:
            ranges = dict()

        if not isinstance(ranges, dict):
            raise ValueError(f"Expected a dict but got a {type(ranges)}")

        cdef std_map[TaskTarget, _ProcessorRange] cpp_ranges
        for target, range in ranges.items():
            if not isinstance(target, PyTaskTarget):
                raise ValueError(f"Invalid task target: {target}")
            if not isinstance(range, ProcessorRange):
                raise ValueError(f"Invalid processor range: {range}")
            cpp_ranges[target] = (<ProcessorRange> range)._handle

        self._handle = _Machine(std_move(cpp_ranges))

    def __len__(self) -> uint32_t:
        """
        Returns the number of preferred processors

        Returns
        -------
        int
            Processor count
        """
        return self.count()

    @property
    def preferred_target(self):
        """
        Returns the preferred kind of processors for mapping tasks

        Returns
        -------
        TaskTarget
            Processor kind
        """
        return self._handle.preferred_target()

    def get_processor_range(
        self, target: Optional[PyTaskTarget] = None
    ) -> ProcessorRange:
        """
        Returns the processor range of a given task target.

        Parameters
        ----------
        target : TaskTarget, optional
            Processor kind to query. If None, the preferred target is used.

        Returns
        -------
        ProcessorRange
            Processor range of the chosen target
        """
        return ProcessorRange.from_handle(
            self._handle.processor_range()
            if target is None
            else self._handle.processor_range(<TaskTarget> target)
        )

    def get_node_range(self, target: Optional[PyTaskTarget] = None) -> tuple:
        """
        Returns the node range for processor of a given task target.

        Parameters
        ----------
        target : TaskTarget, optional
            Processor kind to query. If None, the preferred target is used.

        Returns
        -------
        tuple[int, int]
            Node range for the chosen processor target
        """
        return self.get_processor_range(target).get_node_range()

    @property
    def valid_targets(self):
        """
        Returns the kinds of processors available in this machine

        Returns
        -------
        tuple[TaskTarget, ...]
            Processor kinds
        """
        return tuple(self._handle.valid_targets())

    def count(
        self, target: Optional[PyTaskTarget] = None
    ) -> int:
        """
        Returns the number of processors of a given task target

        Parameters
        ----------
        target : TaskTarget
            Processor kind to query. If None, the preferred target is used.

        Returns
        -------
        int
            Processor count
        """
        return (
            self._handle.count()
            if target is None
            else self._handle.count(<TaskTarget> target)
        )

    @property
    def empty(self):
        """
        Indicates if the machine is empty

        An empty machine is a machine with all its processor ranges being
        empty.

        Returns
        -------
        bool
            ``True`` if the machine is empty, ``False`` otherwise.
        """
        return self._handle.empty()

    def only(
        self, targets: Union[Iterable[TaskTarget], TaskTarget]
    ) -> Machine:
        """
        Returns a machine that contains only the processors of given kinds

        Parameters
        ----------
        targets : TaskTargets
            Kinds of processors to leave in the returned machine

        Returns
        -------
        Machine
            A new machine only with the chosen processors
        """
        if not is_iterable(targets):
            targets = (targets,)
        cdef std_vector[TaskTarget] cpp_targets = std_vector[TaskTarget]()
        for target in targets:
            cpp_targets.push_back(<TaskTarget> target)
        return Machine.from_handle(self._handle.only(std_move(cpp_targets)))

    def slice(
        self, slice sl, target: Optional[PyTaskTarget] = None
    ) -> Machine:
        """
        Slices the machine by a given slice and a task target

        Parameters
        ----------
        sl : slice
            A slice to slice the machine by
        target : TaskTarget, optional
            Processor kind to filter. If None, the preferred target is used.

        Returns
        -------
        Machine
            A new machine after slicing
        """
        if target is None:
            target = self.preferred_target
        elif not isinstance(target, PyTaskTarget):
            raise ValueError(f"Invalid target: {target}")

        return Machine({target: self.get_processor_range(target).slice(sl)})

    def __getitem__(self, key: MACHINE_KEY) -> Machine:
        """
        Slices the machine with a given slicer

        Parameters
        ----------
        key : TaskTarget, slice, int, tuple[TaskTarget, slice]
            Key to slice the machine by

            If the ``key`` is a ``TaskTarget``, a machine with only the
            processors of the chosen kind is returned.

            If the ``key`` is a ``slice``, the returned machine only has a
            processor range for the preferred target, which is sliced by the
            ``key``. An integer ``key`` is treated like a singleton slice
            (i.e., ``slice(key, key + 1)``).

            If the `key` is a pair of a task target and a slice, the
            returned machine only has a processor range of the chosen kind,
            which is sliced by the ``key``.

        Returns
        -------
        Machine
            A new machine after slicing
        """
        if isinstance(key, PyTaskTarget):
            return self.only((key,))
        if isinstance(key, ProcessorSlice):
            return self.slice(key.slice, key.target)
        if isinstance(key, int):
            return self.slice(slice(key, key + 1))
        if isinstance(key, slice):
            return self.slice(key)

        raise KeyError(f"Invalid slicing key: {key}")

    def __eq__(self, other: Machine) -> bool:
        return self._handle == other._handle

    def __ne__(self, other: Machine) -> bool:
        return self._handle != other._handle

    def __and__(self, other: Machine) -> Machine:
        """
        Computes an intersection with a given machine

        Parameters
        ----------
        other : Machine
            A machine to intersect with

        Returns
        -------
        Machine
            Intersection result
        """
        return Machine.from_handle(self._handle & other._handle)

    def __str__(self) -> str:
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        return str(self)

    def __enter__(self) -> None:
        from ..runtime.runtime import get_legate_runtime

        runtime = get_legate_runtime()

        new_machine = runtime.get_machine() & self
        if new_machine.empty:
            raise EmptyMachineError(
                "Empty machines cannot be used for resource scoping"
            )
        runtime.push_machine(new_machine)

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        from ..runtime.runtime import get_legate_runtime

        get_legate_runtime().pop_machine()
