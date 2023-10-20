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

from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.utility cimport move as std_move

from typing import Any, Iterable, Union

from ..data.logical_array cimport LogicalArray, _LogicalArray
from ..data.logical_store cimport LogicalStore, LogicalStorePartition
from ..data.scalar cimport Scalar
from ..partitioning.constraint cimport Constraint, Variable, _align, _broadcast

from ..type.type_info import Type, array_type, null_type

from ..utilities.tuple cimport tuple as _tuple
from ...utils import is_iterable


def sanitized_scalar_arg_type(
    value: Any, dtype: Union[Type, tuple[Type, ...]]
) -> Type:
    sanitized: Type
    if isinstance(dtype, tuple):
        if not (len(dtype) == 1 and isinstance(dtype[0], Type)):
            raise TypeError(f"Unsupported type: {dtype}")
        sanitized = (
            null_type if len(value) == 0 else array_type(dtype[0], len(value))
        )
    elif isinstance(dtype, Type):
        sanitized = dtype
    else:
        raise TypeError(f"Unsupported type: {dtype}")
    return sanitized


cdef _LogicalArray to_logical_array(array_or_store):
    cdef _LogicalArray result
    if isinstance(array_or_store, LogicalArray):
        result = (<LogicalArray> array_or_store)._handle
    elif isinstance(array_or_store, LogicalStore):
        result = _LogicalArray((<LogicalStore> array_or_store)._handle)
    else:
        raise ValueError(
            "Expected a logical array or store but got "
            f"{type(array_or_store)}"
        )
    return result


cdef class AutoTask:
    @staticmethod
    cdef AutoTask from_handle(_AutoTask handle):
        cdef AutoTask result = AutoTask.__new__(AutoTask)
        result._handle = std_move(handle)
        return result

    def add_input(
        self, array_or_store, partition: Union[Variable, None] = None
    ) -> Variable:
        """
        Adds a logical array/store as input to the task

        Parameters
        ----------
        array_or_store : LogicalArray or LogicalStore
            LogicalArray or LogicalStore to pass as input
        partition : Variable, optional
            Partition to associate with the array/store. The default
            partition is picked if none is given.
        """
        cdef _LogicalArray array = to_logical_array(array_or_store)
        if partition is None:
            return Variable.from_handle(self._handle.add_input(array))
        elif isinstance(partition, Variable):
            return Variable.from_handle(
                self._handle.add_input(array, (<Variable> partition)._handle)
            )
        else:
            raise ValueError("Invalid partition symbol")

    def add_output(
        self, array_or_store, partition: Union[Variable, None] = None
    ) -> Variable:
        """
        Adds a logical array/store as output to the task

        Parameters
        ----------
        array_or_store : LogicalArray or LogicalStore
            LogicalArray or LogicalStore to pass as output
        partition : Variable, optional
            Partition to associate with the array/store. The default
            partition is picked if none is given.
        """
        cdef _LogicalArray array = to_logical_array(array_or_store)
        if partition is None:
            return Variable.from_handle(self._handle.add_output(array))
        elif isinstance(partition, Variable):
            return Variable.from_handle(
                self._handle.add_output(array, (<Variable> partition)._handle)
            )
        else:
            raise ValueError("Invalid partition symbol")

    def add_reduction(
        self,
        array_or_store,
        int32_t redop,
        partition: Union[Variable, None] = None
    ) -> Variable:
        """
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
        """
        cdef _LogicalArray array = to_logical_array(array_or_store)
        if partition is None:
            return Variable.from_handle(
                self._handle.add_reduction(array, redop)
            )
        elif isinstance(partition, Variable):
            return Variable.from_handle(
                self._handle.add_reduction(
                    array, redop, (<Variable> partition)._handle
                )
            )
        else:
            raise ValueError("Invalid partition symbol")

    def add_scalar_arg(
        self, value: Any, dtype: Union[Type, tuple[Type, ...], None] = None
    ) -> None:
        """
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
            self._handle.add_scalar_arg((<Scalar> value)._handle)
            return
        if dtype is None:
            raise ValueError(
                "Data type must be given if the value is not a Scalar object"
            )
        sanitized = sanitized_scalar_arg_type(value, dtype)
        cdef Scalar scalar = Scalar(value, sanitized)
        self._handle.add_scalar_arg(scalar._handle)

    def add_constraint(self, Constraint constraint) -> None:
        self._handle.add_constraint(constraint._handle)

    def find_or_declare_partition(self, LogicalArray array) -> Variable:
        return Variable.from_handle(
            self._handle.find_or_declare_partition(array._handle)
        )

    def declare_partition(self) -> Variable:
        return Variable.from_handle(self._handle.declare_partition())

    def provenance(self) -> str:
        return self._handle.provenance().decode()

    def set_concurrent(self, bool concurrent) -> None:
        self._handle.set_concurrent(concurrent)

    def set_side_effect(self, bool has_side_effect) -> None:
        self._handle.set_side_effect(has_side_effect)

    def throws_exception(self, bool can_throw_exception) -> None:
        self._handle.throws_exception(can_throw_exception)

    def add_communicator(self, str name) -> None:
        self._handle.add_communicator(name.encode())

    def execute(self) -> None:
        """
        Submits the operation to the runtime. There is no guarantee that the
        operation will start the execution right upon the return of this
        method.
        """
        from ..runtime.runtime import get_legate_runtime

        get_legate_runtime().submit(self)

    def add_alignment(
        self,
        array_or_store1,
        array_or_store2,
    ) -> None:
        """
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
        array1 = to_logical_array(array_or_store1)
        array2 = to_logical_array(array_or_store2)
        part1 = self._handle.find_or_declare_partition(array1)
        part2 = self._handle.find_or_declare_partition(array2)
        self._handle.add_constraint(_align(part1, part2))

    def add_broadcast(
        self,
        array_or_store,
        axes: Union[None, int, Iterable[int]] = None,
    ) -> None:
        """
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
        array = to_logical_array(array_or_store)
        part = self._handle.find_or_declare_partition(array)

        if axes is None:
            self._handle.add_constraint(_broadcast(part))
            return

        if not (isinstance(axes, int) or is_iterable(axes)):
            raise ValueError("axes must be an integer or an iterable")

        sanitized = (axes,) if isinstance(axes, int) else axes
        if len(sanitized) == 0:
            self._handle.add_constraint(_broadcast(part))
            return

        cdef _tuple[int32_t] cpp_axes
        for axis in sanitized:
            cpp_axes.append_inplace(axis)
        self._handle.add_constraint(_broadcast(part, std_move(cpp_axes)))

    def add_nccl_communicator(self) -> None:
        """
        Adds a NCCL communicator to the task
        """
        self.add_communicator("nccl")

    def add_cpu_communicator(self) -> None:
        """
        Adds a CPU communicator to the task
        """
        self.add_communicator("cpu")


cdef class ManualTask:
    @staticmethod
    cdef ManualTask from_handle(_ManualTask handle):
        cdef ManualTask result = ManualTask.__new__(ManualTask)
        result._handle = std_move(handle)
        return result

    def add_input(
        self, arg: Union[LogicalStore, LogicalStorePartition]
    ) -> None:
        if isinstance(arg, LogicalStore):
            self._handle.add_input((<LogicalStore> arg)._handle)
        elif isinstance(arg, LogicalStorePartition):
            self._handle.add_input((<LogicalStorePartition> arg)._handle)
        else:
            raise ValueError(
                "Expected a logical store or store partition "
                "but got {type(arg)}"
            )

    def add_output(
        self, arg: Union[LogicalStore, LogicalStorePartition]
    ) -> None:
        if isinstance(arg, LogicalStore):
            self._handle.add_output((<LogicalStore> arg)._handle)
        elif isinstance(arg, LogicalStorePartition):
            self._handle.add_output((<LogicalStorePartition> arg)._handle)
        else:
            raise ValueError(
                "Expected a logical store or store partition "
                "but got {type(arg)}"
            )

    def add_reduction(
        self, arg: Union[LogicalStore, LogicalStorePartition], int32_t redop
    ) -> None:
        if isinstance(arg, LogicalStore):
            self._handle.add_reduction((<LogicalStore> arg)._handle, redop)
        elif isinstance(arg, LogicalStorePartition):
            self._handle.add_reduction(
                (<LogicalStorePartition> arg)._handle, redop
            )
        else:
            raise ValueError(
                "Expected a logical store or store partition "
                "but got {type(arg)}"
            )

    def add_scalar_arg(
        self, value: Any, dtype: Union[Type, tuple[Type, ...], None] = None
    ) -> None:
        """
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
            self._handle.add_scalar_arg((<Scalar> value)._handle)
            return
        if dtype is None:
            raise ValueError(
                "Data type must be given if the value is not a Scalar object"
            )
        sanitized = sanitized_scalar_arg_type(value, dtype)
        cdef Scalar scalar = Scalar(value, sanitized)
        self._handle.add_scalar_arg(scalar._handle)

    def provenance(self) -> str:
        return self._handle.provenance().decode()

    def set_concurrent(self, bool concurrent) -> None:
        self._handle.set_concurrent(concurrent)

    def set_side_effect(self, bool has_side_effect) -> None:
        self._handle.set_side_effect(has_side_effect)

    def throws_exception(self, bool can_throw_exception) -> None:
        self._handle.throws_exception(can_throw_exception)

    def add_communicator(self, str name) -> None:
        self._handle.add_communicator(name.encode())

    def execute(self) -> None:
        """
        Submits the operation to the runtime. There is no guarantee that the
        operation will start the execution right upon the return of this
        method.
        """
        from ..runtime.runtime import get_legate_runtime

        get_legate_runtime().submit(self)

    def add_nccl_communicator(self) -> None:
        """
        Adds a NCCL communicator to the task
        """
        self.add_communicator("nccl")

    def add_cpu_communicator(self) -> None:
        """
        Adds a CPU communicator to the task
        """
        self.add_communicator("cpu")
