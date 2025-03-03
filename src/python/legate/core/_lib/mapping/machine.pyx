# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from operator import index as operator_index

from libcpp.map cimport map as std_map
from libcpp.utility cimport move as std_move
from libcpp.vector cimport vector as std_vector

from .mapping cimport TaskTarget

from .mapping import TaskTarget as PyTaskTarget

from ..utilities.utils cimport is_iterable


class EmptyMachineError(Exception):
    pass


@dataclass(frozen=True)
class ProcessorSlice:
    target: TaskTarget
    slice: slice


cdef class ProcessorRange:
    @staticmethod
    def create(
            uint32_t low, uint32_t high, uint32_t per_node_count
    ) -> ProcessorRange:
        r"""
        Construct a `ProcessorRange`.

        Parameters
        ----------
        low : int
            The starting processor ID.
        high : int
            The end processor ID.
        per_node_count : int
            The number of per-node processors.

        Returns
        -------
        ProcessorRange
            The constructed `ProcessorRange`
        """
        return ProcessorRange(low, high, per_node_count)

    @staticmethod
    def create_empty() -> ProcessorRange:
        r"""
        Create an empty `ProcessorRange`.

        Returns
        -------
        ProcessorRange
            The empty range.

        Notes
        -----
        Equivalent to `ProcessorRange()` (i.e. an empty constructor).
        """
        return ProcessorRange()

    @staticmethod
    cdef ProcessorRange from_handle(_ProcessorRange handle):
        cdef ProcessorRange result = ProcessorRange.__new__(ProcessorRange)
        result._handle = std_move(handle)
        return result

    def __cinit__(
        self,
        uint32_t low = 0,
        uint32_t high = 0,
        uint32_t per_node_count = 0,
    ) -> None:
        r"""
        Construct a `ProcessorRange`.

        Parameters
        ----------
        low : int
            The starting processor ID.
        high : int
            The end processor ID.
        per_node_count : int
            The number of per-node processors.
        """
        self._handle = _ProcessorRange(low, high, per_node_count)

    @property
    def low(self) -> uint32_t:
        r"""
        Returns the lower bound of the processor range

        :returns: Lower bound (inclusive)
        :rtype: int
        """
        return self._handle.low

    @property
    def high(self) -> uint32_t:
        r"""
        Returns the upper bound of the processor range

        :returns: Upper bound (exclusive)
        :rtype: int
        """
        return self._handle.high

    @property
    def per_node_count(self) -> uint32_t:
        r"""
        Returns the number of processors per node

        :returns: Per-node processor count
        :rtype: int
        """
        return self._handle.per_node_count

    @property
    def count(self) -> uint32_t:
        r"""
        Returns the number of processors in the range

        :returns: Processor count
        :rtype: int
        """
        return self._handle.count()

    def __len__(self) -> uint32_t:
        r"""
        Returns the number of processors in the range

        :returns: Processor count
        :rtype: int
        """
        return self.count

    @property
    def empty(self) -> bool:
        r"""
        Indicates if the processor range is empty

        :returns: ``True`` if the machine is empty, ``False`` otherwise.
        :rtype: bool
        """
        return self._handle.empty()

    cpdef ProcessorRange slice(self, slice sl):
        r"""
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
        cdef int start = 0 if sl.start is None else sl.start
        cdef int stop = self.high if sl.stop is None else sl.stop
        return ProcessorRange.from_handle(self._handle.slice(start, stop))

    def __getitem__(self, key: slice | int) -> ProcessorRange:
        r"""
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
        if isinstance(key, slice):
            return self.slice(key)
        key = operator_index(key)
        return self.slice(slice(key, key + 1))

    cpdef tuple get_node_range(self):
        r"""
        Returns the range of node IDs for this processor range

        Returns
        -------
        tuple[int, int]
            Half-open interval of node IDs
        """
        result = self._handle.get_node_range()
        return (result.low, result.high)

    def __str__(self) -> str:
        r"""
        Return a human-readable representation of the range.

        Returns
        -------
        str
            The human readable representation of the range.
        """
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        r"""
        Return a human-readable representation of the range.

        Returns
        -------
        str
            The human readable representation of the range.
        """
        return str(self)

    def __and__(self, other: ProcessorRange) -> ProcessorRange:
        r"""
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
        if isinstance(other, ProcessorRange):
            return self._handle == other._handle
        return NotImplemented

    def __ne__(self, other: ProcessorRange) -> bool:
        return not (self._handle == other._handle)

    def __lt__(self, other: ProcessorRange) -> bool:
        if isinstance(other, ProcessorRange):
            return self._handle < other._handle
        return NotImplemented


cdef class Machine:
    @staticmethod
    cdef Machine from_handle(_Machine handle):
        cdef Machine result = Machine.__new__(Machine)
        result._handle = std_move(handle)
        result._scope = None
        return result

    def __cinit__(self, ranges: dict | None = None) -> None:
        r"""
        Construct a `Machine`.

        Parameters
        ----------
        ranges : dict[TaskTarget, ProcessorRange] | None
            A mapping of the avaible processors per target for the machine.

        Raises
        ------
        ValueError
            If `ranges` is neither `None` nor a `dict`.
        ValueError
            If a key in `ranges` is not a `TaskTarget`.
        ValueError
            If a value in `ranges` is not a `ProcessorRange`.
        """
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
        self._scope = None

    def __len__(self) -> uint32_t:
        r"""
        Returns the number of preferred processors

        Returns
        -------
        int
            Processor count
        """
        return self.count()

    @property
    def preferred_target(self) -> TaskTarget:
        r"""
        Returns the preferred kind of processors for mapping tasks

        :returns: Processor kind
        :rtype: TaskTarget
        """
        cdef TaskTarget ret

        with nogil:
            ret = self._handle.preferred_target()
        return ret

    cpdef ProcessorRange get_processor_range(
        self, target: PyTaskTarget | None = None
    ):
        r"""
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

    cpdef tuple get_node_range(self, target: PyTaskTarget | None = None):
        r"""
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
    def valid_targets(self) -> tuple[TaskTarget, ...]:
        r"""
        Returns the kinds of processors available in this machine

        :returns: Processor kinds
        :rtype: tuple[TaskTarget, ...]
        """
        cdef const std_vector[TaskTarget] *v = NULL

        with nogil:
            v = &self._handle.valid_targets()
        return tuple(v[0])

    cpdef int count(
        self, target: PyTaskTarget | None = None
    ):
        r"""
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
        cdef int ret
        cdef TaskTarget tgt

        if target is None:
            with nogil:
                ret = self._handle.count()
        else:
            tgt = <TaskTarget> target
            with nogil:
                ret = self._handle.count(tgt)
        return ret

    @property
    def empty(self) -> bool:
        r"""
        Indicates if the machine is empty

        An empty machine is a machine with all its processor ranges being
        empty.

        :returns: ``True`` if the machine is empty, ``False`` otherwise.
        :rtype: bool
        """
        cdef bool ret

        with nogil:
            ret = self._handle.empty()
        return ret

    cpdef Machine only(
        self, targets: Iterable[TaskTarget] | TaskTarget
    ):
        r"""
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

        cpp_targets.reserve(len(targets))
        for target in targets:
            cpp_targets.push_back(<TaskTarget> target)

        cdef _Machine handle

        with nogil:
            handle = self._handle.only(std_move(cpp_targets))
        return Machine.from_handle(std_move(handle))

    cpdef Machine slice(
        self, slice sl, target: PyTaskTarget | None = None
    ):
        r"""
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

    def __getitem__(
        self, key: PyTaskTarget | slice | int | ProcessorSlice
    ) -> Machine:
        r"""
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
        if isinstance(key, slice):
            return self.slice(key)
        key = operator_index(key)
        return self.slice(slice(key, key + 1))

    def __eq__(self, other: Machine) -> bool:
        cdef bool ret

        if isinstance(other, Machine):
            with nogil:
                ret = self._handle == other._handle
            return ret
        return NotImplemented

    def __ne__(self, other: Machine) -> bool:
        return not self == other

    def __and__(self, other: Machine) -> Machine:
        r"""
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
        cdef _Machine handle

        with nogil:
            handle = self._handle & other._handle
        return Machine.from_handle(std_move(handle))

    def __str__(self) -> str:
        r"""
        Return a human-readable representation of the `Machine`.

        Returns
        -------
        str
            The human readable representation of the `Machine`.
        """
        return self._handle.to_string().decode()

    def __repr__(self) -> str:
        r"""
        Return a human-readable representation of the `Machine`.

        Returns
        -------
        str
            The human readable representation of the `Machine`.
        """
        return str(self)

    def __enter__(self) -> None:
        if self._scope is not None:
            raise ValueError("Each machine can be set only once to the scope")
        self._scope = Scope(machine=self)
        try:
            self._scope.__enter__()
        except RuntimeError as e:
            self._scope = None
            raise EmptyMachineError(str(e))

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        self._scope.__exit__(_, __, ___)
        self._scope = None
