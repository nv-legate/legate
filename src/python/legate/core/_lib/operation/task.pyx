# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int32_t, uint32_t, uintptr_t
from libcpp cimport bool
from libcpp.optional cimport make_optional, optional as std_optional
from libcpp.utility cimport move as std_move

from ..._ext.cython_libcpp.string_view cimport (
    str_from_string_view,
    string_view_from_py as std_string_view_from_py,
)

from collections.abc import Iterable
from typing import Any

from ..data.logical_array cimport (
    LogicalArray,
    _LogicalArray,
    to_cpp_logical_array,
)
from ..data.logical_store cimport LogicalStore, LogicalStorePartition
from ..data.scalar cimport Scalar
from ..partitioning.constraint cimport Constraint, Variable, _align, _broadcast
from ..runtime.runtime cimport get_legate_runtime
from ..type.type_info cimport Type, array_type
from ..utilities.unconstructable cimport Unconstructable
from ..utilities.utils cimport is_iterable
from .projection cimport SymbolicExpr, _SymbolicPoint

from ..type.type_info import null_type

from ..utilities.tuple cimport _tuple


cdef Type sanitized_scalar_arg_type(
    object value, dtype: Type | tuple[Type, ...]
):
    if isinstance(dtype, Type):
        return dtype

    if isinstance(dtype, tuple):
        if not (len(dtype) == 1 and isinstance(dtype[0], Type)):
            raise TypeError(f"Unsupported type: {dtype}")
        return (
            null_type if len(value) == 0 else array_type(dtype[0], len(value))
        )

    raise TypeError(f"Unsupported type: {dtype}")


cdef class AutoTask(Unconstructable):
    @staticmethod
    cdef AutoTask from_handle(_AutoTask handle):
        cdef AutoTask result = AutoTask.__new__(AutoTask)
        result._handle = std_move(handle)
        result._exception_types = dict()
        result._locked = False
        return result

    # TODO(jfaibussowit): consider removing this, see
    # https://github.com/nv-legate/legate.core.internal/pull/309
    cpdef void lock(self):
        r"""Lock an `AutoTask` from further argument modifications.

        Notes
        -----
        This prevents any additional modification of the inputs, outputs,
        reductions, or scalar arguments to this particular AutoTask. This is
        usually the final action performed by a ``PyTask``.
        """
        self._locked = True

    cpdef Variable add_input(
        self, object array_or_store, partition: Variable | None = None
    ):
        r"""
        Adds a logical array/store as input to the task

        Parameters
        ----------
        array_or_store : LogicalArray or LogicalStore
            LogicalArray or LogicalStore to pass as input
        partition : Variable, optional
            Partition to associate with the array/store. The default
            partition is picked if none is given.

        Raises
        ------
        RuntimeError
            If the AutoTask has been previously locked by a ``PyTask``.
        """
        if self._locked:
            raise RuntimeError(
                "Attempting to add inputs to a prepared Python task "
                "is illegal!"
            )

        cdef _LogicalArray array = to_cpp_logical_array(array_or_store)
        cdef _Variable handle

        if partition is None:
            with nogil:
                handle = self._handle.add_input(array)
            return Variable.from_handle(std_move(handle))
        elif isinstance(partition, Variable):
            with nogil:
                handle = self._handle.add_input(
                    array, (<Variable> partition)._handle
                )
            return Variable.from_handle(std_move(handle))
        else:
            raise ValueError("Invalid partition symbol")

    cpdef Variable add_output(
        self, object array_or_store, partition: Variable | None = None
    ):
        r"""
        Adds a logical array/store as output to the task

        Parameters
        ----------
        array_or_store : LogicalArray or LogicalStore
            LogicalArray or LogicalStore to pass as output
        partition : Variable, optional
            Partition to associate with the array/store. The default
            partition is picked if none is given.

        Raises
        ------
        RuntimeError
            If the AutoTask has been previously locked by a `PyTask`.
        """
        if self._locked:
            raise RuntimeError(
                "Attempting to add outputs to a prepared Python task "
                "is illegal!"
            )

        cdef _LogicalArray array = to_cpp_logical_array(array_or_store)
        cdef _Variable handle

        if partition is None:
            with nogil:
                handle = self._handle.add_output(array)
        elif isinstance(partition, Variable):
            with nogil:
                handle = self._handle.add_output(
                    array, (<Variable> partition)._handle
                )
        else:
            raise ValueError("Invalid partition symbol")

        return Variable.from_handle(std_move(handle))

    cpdef Variable add_reduction(
        self,
        object array_or_store,
        int32_t redop,
        partition: Variable | None = None
    ):
        r"""
        Adds a logical array/store to the task for reduction

        Parameters
        ----------
        array_or_store : LogicalArray or LogicalStore
            LogicalArray or LogicalStore to pass for reduction
        redop : int
            Reduction operator ID
        partition : Variable, optional
            Partition to associate with the array/store. The default
            partition is picked if none is given.

        Raises
        ------
        RuntimeError
            If the AutoTask has been previously locked by a `PyTask`.
        """
        if self._locked:
            raise RuntimeError(
                "Attempting to add reductions to a prepared Python task "
                "is illegal!"
            )

        cdef _LogicalArray array = to_cpp_logical_array(array_or_store)
        cdef _Variable handle

        if partition is None:
            with nogil:
                handle = self._handle.add_reduction(array, redop)
        elif isinstance(partition, Variable):
            with nogil:
                handle = self._handle.add_reduction(
                    array, redop, (<Variable> partition)._handle
                )
        else:
            raise ValueError("Invalid partition symbol")

        return Variable.from_handle(std_move(handle))

    cpdef void add_scalar_arg(
        self, value: Any, dtype: Type | tuple[Type, ...] | None = None
    ):
        r"""
        Adds a by-value argument to the task

        Parameters
        ----------
        value : Any
            Scalar value or a tuple of scalars (but no nested tuples)
        dtype : Dtype
            Data type descriptor for the scalar value. A descriptor ``(T,)``
            means that the value is a tuple of elements of type ``T`` (i.e.,
            equivalent to ``array_type(T, len(value))``).

        Raises
        ------
        RuntimeError
            If the AutoTask has been previously locked by a `PyTask`.
        """
        if self._locked:
            raise RuntimeError(
                "Attempting to add scalar arguments to a prepared Python task "
                "is illegal!"
            )

        if isinstance(value, Scalar):
            with nogil:
                self._handle.add_scalar_arg((<Scalar> value)._handle)
            return

        cdef Scalar scalar

        if dtype is None:
            scalar = Scalar(value)
        else:
            scalar = Scalar(
                value, dtype=sanitized_scalar_arg_type(value, dtype)
            )
        with nogil:
            self._handle.add_scalar_arg(scalar._handle)

    cpdef void add_constraint(self, Constraint constraint):
        r"""
        Add a partitioning constraint to the task.

        Parameters
        ----------
        constraint : Constraint
            The partitioning constraint to add.
        """
        with nogil:
            self._handle.add_constraint(constraint._handle)

    cpdef Variable find_or_declare_partition(self, LogicalArray array):
        r"""
        Finds or creates a partition symbol for the given array.

        Parameters
        ----------
        array : LogicalArray
            The array for which to look for.

        Returns
        -------
        Variable
            The partition symbol.
        """
        cdef _Variable handle

        with nogil:
            handle = self._handle.find_or_declare_partition(array._handle)
        return Variable.from_handle(std_move(handle))

    cpdef Variable declare_partition(self):
        r"""
        Declare a partition symbol for this task.

        Returns
        -------
        Variable
            The partition symbol.

        Notes
        -----
        As opposed to `find_or_declare_partition()`, this routine always
        returns a new `Variable`.
        """
        cdef _Variable handle

        with nogil:
            handle = self._handle.declare_partition()
        return Variable.from_handle(std_move(handle))

    cpdef str provenance(self):
        r"""
        Returns the provenance of this task.

        Returns
        -------
        str
            The provenance.
        """
        return str_from_string_view(self._handle.provenance())

    cpdef void set_concurrent(self, bool concurrent):
        r"""
        Set whether a task requires a concurrent task launch.

        Any task with at least one communicator will implicitly use concurrent
        task launch, so this method is to be used when the task needs a
        concurrent task launch for a reason unknown to Legate.

        All tasks -- unless specified by this routine, or by the implicit
        condition above -- default to non-concurrent task launch.

        Parameters
        ----------
        concurrent : bool
            `True` if the task should be concurrent, `False` otherwise.
        """
        with nogil:
            self._handle.set_concurrent(concurrent)

    cpdef void set_side_effect(self, bool has_side_effect):
        r"""
        Set whether a task has side effects.

        A task is assumed to be free of side effects by default if the task
        only has scalar arguments.

        Parameters
        ----------
        has_side_effect : bool
            `True` if the task has side effects, `False` otherwise.
        """
        with nogil:
            self._handle.set_side_effect(has_side_effect)

    cpdef void throws_exception(self, type exception_type):
        r"""
        Set which exception is thrown by the task.

        This routine may be called multiple times to indicate that the task
        throws multiple exceptions. Each exception is added to the set of
        thrown exceptions. The exception types are deduplicated.

        Parameters
        ----------
        exception_type : type
            The type of exception thrown by the task.
        """
        with nogil:
            self._handle.throws_exception(True)
        self._exception_types[exception_type] = None

    @property
    def exception_types(self) -> tuple[type, ...]:
        r"""
        Get the exception types thrown by the task.

        :returns: The types of exceptions thrown by the task.
        :rtype: tuple[type, ...]
        """
        return tuple(self._exception_types.keys())

    cpdef void add_communicator(self, str name):
        r"""
        Add a communicator to the task

        Parameters
        ----------
        name : str
            The name of the communicator to add.
        """
        self._handle.add_communicator(std_string_view_from_py(name))

    cpdef void execute(self):
        r"""
        Submits the operation to the runtime.

        There is no guarantee of when the runtime will start the execution of
        the task.
        """
        get_legate_runtime().submit(self)

    cpdef void add_alignment(
        self,
        object array_or_store1,
        object array_or_store2,
    ):
        r"""
        Sets an alignment between stores. Equivalent to the following code:

        ::

            symb1 = op.declare_partition(store1)
            symb2 = op.declare_partition(store2)
            op.add_constraint(symb1 == symb2)

        Parameters
        ----------
        arr_or_store1, arr_or_store2 : LogicalArray or LogicalStore
            LogicalArrays or LogicalStores to align

        Raises
        ------
        ValueError
            If the stores don't have the same shape or only one of them is
            unbound
        """
        array1 = to_cpp_logical_array(array_or_store1)
        array2 = to_cpp_logical_array(array_or_store2)
        with nogil:
            part1 = self._handle.find_or_declare_partition(array1)
            part2 = self._handle.find_or_declare_partition(array2)
            self._handle.add_constraint(_align(part1, part2))

    cpdef void add_broadcast(
        self,
        object array_or_store,
        axes: int | Iterable[int] | None = None,
    ):
        r"""
        Sets a broadcasting constraint on the logical_array. Equivalent to the
        following code:

        ::

            symb = op.declare_partition(logical_array)
            op.add_constraint(symb.broadcast(axes))

        Parameters
        ----------
        array_or_store : LogicalArray or LogicalStore
            LogicalArray or LogicalStore to set a broadcasting constraint on
        axes : int or Iterable[int], optional
            Axes to broadcast. The entire logical_array is
            replicated if no axes are given.
        """
        array = to_cpp_logical_array(array_or_store)
        part = self._handle.find_or_declare_partition(array)

        if axes is None:
            with nogil:
                self._handle.add_constraint(_broadcast(part))
            return

        if not (isinstance(axes, int) or is_iterable(axes)):
            raise ValueError("axes must be an integer or an iterable")

        sanitized = (axes,) if isinstance(axes, int) else axes
        if len(sanitized) == 0:
            with nogil:
                self._handle.add_constraint(_broadcast(part))
            return

        cdef _tuple[uint32_t] cpp_axes
        cpp_axes.reserve(len(sanitized))
        for axis in sanitized:
            cpp_axes.append_inplace(axis)
        with nogil:
            self._handle.add_constraint(_broadcast(part, std_move(cpp_axes)))

    cpdef void add_nccl_communicator(self):
        r"""
        Adds a NCCL communicator to the task
        """
        self.add_communicator("nccl")

    cpdef void add_cpu_communicator(self):
        r"""
        Adds a CPU communicator to the task
        """
        self.add_communicator("cpu")

    cpdef void add_cal_communicator(self):
        r"""
        Adds a CAL communicator to the task
        """
        self.add_communicator("cal")

    @property
    def raw_handle(self) -> uintptr_t:
        r"""
        Get the raw C++ pointer to the underlying class instance as an integer

        :returns: The pointer to the C++ `AutoTask`.
        :rtype: int
        """
        return <uintptr_t> &self._handle


cdef std_optional[_SymbolicPoint] to_cpp_projection(object projection):
    if projection is None:
        return std_optional[_SymbolicPoint]()
    if not isinstance(projection, tuple):
        raise ValueError(f"Expected a tuple, but got {type(projection)}")
    cdef _SymbolicPoint result
    for expr in projection:
        result.append_inplace((<SymbolicExpr> expr)._handle)
    return make_optional[_SymbolicPoint](std_move(result))


cdef class ManualTask(Unconstructable):
    @staticmethod
    cdef ManualTask from_handle(_ManualTask handle):
        cdef ManualTask result = ManualTask.__new__(ManualTask)
        result._handle = std_move(handle)
        result._exception_types = dict()
        return result

    cpdef void add_input(
        self,
        arg: LogicalStore | LogicalStorePartition,
        projection: tuple[SymbolicExpr, ...] | None = None,
    ):
        r"""
        Adds an input to the task.

        Parameters
        ----------
        arg : LogicalStore | LogicalStorePartition
            `LogicalStore` or `LogicalStorePartition` to pass as input.
        projection : tuple[SymbolicExpr, ...] | None
            The projection for the partition (if `arg` is a
            `LogicalStorePartition`). If `arg` is a `LogicalStore`, then
            argument is ignored.

        Raises
        ------
        TypeError
            If `arg` is neither a `LogicalStore` or `LogicalStorePartition`.
        """
        cdef std_optional[_SymbolicPoint] proj

        if isinstance(arg, LogicalStore):
            with nogil:
                self._handle.add_input((<LogicalStore> arg)._handle)
        elif isinstance(arg, LogicalStorePartition):
            proj = to_cpp_projection(projection)
            with nogil:
                self._handle.add_input(
                    (<LogicalStorePartition> arg)._handle,
                    std_move(proj),
                )
        else:
            raise TypeError(
                "Expected a logical store or store partition "
                "but got {type(arg)}"
            )

    cpdef void add_output(
        self,
        arg: LogicalStore | LogicalStorePartition,
        projection: tuple[SymbolicExpr, ...] | None = None,
    ):
        r"""
        Adds an output to the task.

        Parameters
        ----------
        arg : LogicalStore | LogicalStorePartition
            `LogicalStore` or `LogicalStorePartition` to pass as output.
        projection : tuple[SymbolicExpr, ...] | None
            The projection for the partition (if `arg` is a
            `LogicalStorePartition`). If `arg` is a `LogicalStore`, then
            argument is ignored.

        Raises
        ------
        TypeError
            If `arg` is neither a `LogicalStore` or `LogicalStorePartition`.
        """
        cdef std_optional[_SymbolicPoint] proj

        if isinstance(arg, LogicalStore):
            with nogil:
                self._handle.add_output((<LogicalStore> arg)._handle)
        elif isinstance(arg, LogicalStorePartition):
            proj = to_cpp_projection(projection)
            with nogil:
                self._handle.add_output(
                    (<LogicalStorePartition> arg)._handle,
                    std_move(proj),
                )
        else:
            raise TypeError(
                "Expected a logical store or store partition "
                "but got {type(arg)}"
            )

    cpdef void add_reduction(
        self,
        arg: LogicalStore | LogicalStorePartition,
        int32_t redop,
        projection: tuple[SymbolicExpr, ...] | None = None,
    ):
        cdef std_optional[_SymbolicPoint] proj

        if isinstance(arg, LogicalStore):
            with nogil:
                self._handle.add_reduction((<LogicalStore> arg)._handle, redop)
        elif isinstance(arg, LogicalStorePartition):
            proj = to_cpp_projection(projection)
            with nogil:
                self._handle.add_reduction(
                    (<LogicalStorePartition> arg)._handle,
                    redop,
                    std_move(proj)
                )
        else:
            raise ValueError(
                "Expected a logical store or store partition "
                "but got {type(arg)}"
            )

    cpdef void add_scalar_arg(
        self, value: Any, dtype: Type | tuple[Type, ...] | None = None
    ):
        r"""
        Adds a by-value argument to the task

        Parameters
        ----------
        value : Any
            Scalar value or a tuple of scalars (but no nested tuples)
        dtype : Dtype
            Data type descriptor for the scalar value. A descriptor ``(T,)``
            means that the value is a tuple of elements of type ``T`` (i.e.,
            equivalent to ``array_type(T, len(value))``).
        """
        if isinstance(value, Scalar):
            with nogil:
                self._handle.add_scalar_arg((<Scalar> value)._handle)
            return

        cdef Scalar scalar

        if dtype is None:
            scalar = Scalar(value)
        else:
            scalar = Scalar(
                value, dtype=sanitized_scalar_arg_type(value, dtype)
            )

        with nogil:
            self._handle.add_scalar_arg(scalar._handle)

    cpdef str provenance(self):
        r"""
        Returns the provenance of this task.

        Returns
        -------
        str
            The provenance.
        """
        return str_from_string_view(self._handle.provenance())

    cpdef void set_concurrent(self, bool concurrent):
        r"""
        Set whether a task requires a concurrent task launch.

        Any task with at least one communicator will implicitly use concurrent
        task launch, so this method is to be used when the task needs a
        concurrent task launch for a reason unknown to Legate.

        All tasks -- unless specified by this routine, or by the implicit
        condition above -- default to non-concurrent task launch.

        Parameters
        ----------
        concurrent : bool
            `True` if the task should be concurrent, `False` otherwise.
        """
        with nogil:
            self._handle.set_concurrent(concurrent)

    cpdef void set_side_effect(self, bool has_side_effect):
        r"""
        Set whether a task has side effects.

        A task is assumed to be free of side effects by default if the task
        only has scalar arguments.

        Parameters
        ----------
        has_side_effect : bool
            `True` if the task has side effects, `False` otherwise.
        """
        with nogil:
            self._handle.set_side_effect(has_side_effect)

    cpdef void throws_exception(self, type exception_type):
        r"""
        Set which exception is thrown by the task.

        This routine may be called multiple times to indicate that the task
        throws multiple exceptions. Each exception is added to the set of
        thrown exceptions. The exception types are deduplicated.

        Parameters
        ----------
        exception_type : type
            The type of exception thrown by the task.
        """
        with nogil:
            self._handle.throws_exception(True)
        self._exception_types[exception_type] = None

    @property
    def exception_types(self) -> tuple[type]:
        r"""
        Get the exception types thrown by the task.

        :returns: The types of exceptions thrown by the task.
        :rtype: tuple[type, ...]
        """
        return tuple(self._exception_types.keys())

    cpdef void add_communicator(self, str name):
        r"""
        Add a communicator to the task

        Parameters
        ----------
        name : str
            The name of the communicator to add.
        """
        self._handle.add_communicator(std_string_view_from_py(name))

    cpdef void execute(self):
        r"""
        Submits the operation to the runtime.

        There is no guarantee when the runtime will start the execution of this
        task.
        """
        get_legate_runtime().submit(self)

    cpdef void add_nccl_communicator(self):
        r"""
        Adds a NCCL communicator to the task
        """
        self.add_communicator("nccl")

    cpdef void add_cpu_communicator(self):
        r"""
        Adds a CPU communicator to the task
        """
        self.add_communicator("cpu")

    cpdef void add_cal_communicator(self):
        r"""
        Adds a CAL communicator to the task
        """
        self.add_communicator("cal")

    @property
    def raw_handle(self) -> uintptr_t:
        r"""
        Get the raw C++ pointer to the underlying class instance as an integer.

        :returns: The raw pointer to the C++ `ManualTask`.
        :rtype: int
        """
        return <uintptr_t> &self._handle
