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

from cpython.buffer cimport (
    Py_buffer,
    PyBUF_CONTIG,
    PyBUF_CONTIG_RO,
    PyObject_GetBuffer,
)
from libc.stdint cimport int32_t, int64_t, uint32_t
from libcpp cimport bool
from libcpp.utility cimport move as std_move

import gc
import inspect
import sys
from typing import Any, Iterable, Optional

import cython
from cython.cimports.libc.stdlib import malloc

from legion_top import add_cleanup_item

from ..data.logical_array cimport (
    LogicalArray,
    _LogicalArray,
    to_cpp_logical_array,
)
from ..data.logical_store cimport LogicalStore
from ..data.scalar cimport Scalar
from ..data.shape cimport _Shape
from ..mapping.machine cimport Machine
from ..operation.task cimport AutoTask, ManualTask
from ..type.type_info cimport Type
from .library cimport Library
from .tracker cimport _ProvenanceTracker

from ...shape import Shape
from ...utils import AnyCallable, ShutdownCallback, is_iterable, is_shape_like


class ShutdownCallbackManager:
    def __init__(self) -> None:
        self._shutdown_callbacks: list[ShutdownCallback] = []

    def add_shutdown_callback(self, callback: ShutdownCallback) -> None:
        self._shutdown_callbacks.append(callback)

    def perform_callbacks(self) -> None:
        for callback in self._shutdown_callbacks:
            callback()


_shutdown_manager = ShutdownCallbackManager()


create_legate_task_exception()
LegateTaskException = <object> _LegateTaskException


cdef void _reraise_legate_exception(tuple[type] exception_types, Exception e):
    message, index = e.args
    try:
        exn = exception_types[index]
    except IndexError:
        raise RuntimeError(f"Invalid exception index {index}")

    raise exn(message)


cdef class Runtime:
    @staticmethod
    cdef Runtime from_handle(_Runtime* handle):
        cdef Runtime result = Runtime.__new__(Runtime)
        result._handle = handle
        return result

    def find_library(self, str library_name) -> Library:
        return Library.from_handle(
            self._handle.find_library(library_name.encode())
        )

    @property
    def core_library(self) -> Library:
        return self.find_library("legate.core")

    def create_auto_task(self, Library library, int64_t task_id) -> AutoTask:
        """
        Creates an auto task.

        Parameters
        ----------
        library: Library
            Library to which the task id belongs

        task_id : int
            Task id. Scoped locally within the library; i.e., different
            libraries can use the same task id. There must be a task
            implementation corresponding to the task id.

        Returns
        -------
        AutoTask
            A new automatically parallelized task
        """
        return AutoTask.from_handle(
            self._handle.create_task(library._handle, task_id)
        )

    def create_manual_task(
        self,
        Library library,
        int64_t task_id,
        launch_shape,
    ) -> ManualTask:
        """
        Creates a manual task.

        Parameters
        ----------
        library: Library
            Library to which the task id belongs

        task_id : int
            Task id. Scoped locally within the library; i.e., different
            libraries can use the same task id. There must be a task
            implementation corresponding to the task id.

        launch_domain : tuple, optional
            Launch domain of the task.

        Returns
        -------
        ManualTask
            A new task
        """
        if not is_iterable(launch_shape):
            raise ValueError("Launch space must be iterable")

        cdef _Shape shape = _Shape()
        for value in launch_shape:
            shape.append_inplace(<size_t>value)

        return ManualTask.from_handle(
            self._handle.create_task(library._handle, task_id, shape)
        )

    def issue_copy(
        self,
        LogicalStore target,
        LogicalStore source,
        redop: Optional[int] = None,
    ) -> None:
        if redop is None:
            self._handle.issue_copy(target._handle, source._handle)
        else:
            self._handle.issue_copy(
                target._handle, source._handle, <int32_t> redop
            )

    def issue_gather(
        self,
        LogicalStore target,
        LogicalStore source,
        LogicalStore source_indirect,
        redop: Optional[int] = None,
    ) -> None:
        if redop is None:
            self._handle.issue_gather(
                target._handle,
                source._handle,
                source_indirect._handle,
            )
        else:
            self._handle.issue_gather(
                target._handle,
                source._handle,
                source_indirect._handle,
                <int32_t> redop,
            )

    def issue_scatter(
        self,
        LogicalStore target,
        LogicalStore target_indirect,
        LogicalStore source,
        redop: Optional[int] = None,
    ) -> None:
        if redop is None:
            self._handle.issue_scatter(
                target._handle,
                target_indirect._handle,
                source._handle,
            )
        else:
            self._handle.issue_scatter(
                target._handle,
                target_indirect._handle,
                source._handle,
                <int32_t> redop,
            )

    def issue_scatter_gather(
        self,
        LogicalStore target,
        LogicalStore target_indirect,
        LogicalStore source,
        LogicalStore source_indirect,
        redop: Optional[int] = None,
    ) -> None:
        if redop is None:
            self._handle.issue_scatter_gather(
                target._handle,
                target_indirect._handle,
                source._handle,
                source_indirect._handle,
            )
        else:
            self._handle.issue_scatter_gather(
                target._handle,
                target_indirect._handle,
                source._handle,
                source_indirect._handle,
                <int32_t> redop,
            )

    def issue_fill(self, object array_or_store, object value) -> None:
        """
        Fills the array or store with a constant value.

        Parameters
        ----------
        array_or_store : LogicalArray or LogicalStore
            LogicalArray or LogicalStore to fill

        value : LogicalStore or Scalar
            The constant value to fill the ``array_or_store`` with

        Raises
        ------
        ValueError
            Any of the following cases:
            1) ``array_or_store`` is not a ``LogicalArray`` or
               a ``LogicalStore``
            2) ``array_or_store`` is unbound
            3) ``value`` is not a ``Scalar`` or a scalar ``LogicalStore``
            or the ``array_or_store`` is unbound
        """
        cdef _LogicalArray arr = to_cpp_logical_array(array_or_store)
        if isinstance(value, LogicalStore):
            self._handle.issue_fill(
                arr, (<LogicalStore> value)._handle
            )
        elif isinstance(value, Scalar):
            self._handle.issue_fill(arr, (<Scalar> value)._handle)
        else:
            raise ValueError(
                "Fill value must be a logical store or a scalar but "
                f"got {type(value)}"
            )

    def tree_reduce(
        self,
        Library library,
        int64_t task_id,
        LogicalStore store,
        int64_t radix = 4,
    ) -> LogicalStore:
        """
        Performs a user-defined reduction by building a tree of reduction
        tasks. At each step, the reducer task gets up to ``radix`` input stores
        and is supposed to produce outputs in a single unbound store.

        Parameters
        ----------
        library: Library
            Library to which the task id belongs

        task_id : int
            Id of the reducer task

        store : LogicalStore
            LogicalStore to perform reductions on

        radix : int
            Fan-in of each reducer task. If the store is partitioned into
            :math:`N` sub-stores by the runtime, then the first level of
            reduction tree has :math:`\\ceil{N / \\mathtt{radix}}` reducer
            tasks.

        Returns
        -------
        LogicalStore
            LogicalStore that contains reduction results
        """
        return LogicalStore.from_handle(
            self._handle.tree_reduce(
                library._handle, task_id, store._handle, radix
            )
        )

    def submit(self, op) -> None:
        if isinstance(op, AutoTask):
            try:
                self._handle.submit((<AutoTask> op)._handle)
                return
            except LegateTaskException as e:
                _reraise_legate_exception(op.exception_types, e)
        elif isinstance(op, ManualTask):
            try:
                self._handle.submit((<ManualTask> op)._handle)
                return
            except LegateTaskException as e:
                _reraise_legate_exception(op.exception_types, e)

        raise RuntimeError(f"Unknown type of operation: {type(op)}")

    def create_array(
        self,
        Type dtype,
        shape: Optional[Union[Shape, Iterable[int]]] = None,
        bool nullable = False,
        bool optimize_scalar = False,
        ndim: Optional[int] = None,
    ) -> LogicalArray:
        if ndim is not None and shape is not None:
            raise ValueError("ndim cannot be used with shape")

        if ndim is None and shape is None:
            ndim = 1

        if shape is None:
            return LogicalArray.from_handle(
                self._handle.create_array(dtype._handle, ndim, nullable)
            )

        if not is_shape_like(shape):
            raise ValueError(
                "shape must be a Shape object or an iterable"
            )

        cdef _Shape cpp_shape = _Shape()
        for extent in shape:
            try:
                cpp_shape.append_inplace(<size_t> extent)
            except OverflowError:
                raise ValueError("Extent must be a positive number")

        return LogicalArray.from_handle(
            self._handle.create_array(
                std_move(cpp_shape), dtype._handle, nullable, optimize_scalar
            )
        )

    def create_array_like(
        self, LogicalArray array, Type dtype
    ) -> LogicalArray:
        return LogicalArray.from_handle(
            self._handle.create_array_like(
                array._handle, dtype._handle
            )
        )

    def create_store(
        self,
        Type dtype,
        shape: Optional[Union[Shape, Iterable[int]]] = None,
        bool optimize_scalar = False,
        ndim: Optional[int] = None,
    ) -> LogicalStore:
        if ndim is not None and shape is not None:
            raise ValueError("ndim cannot be used with shape")

        if shape is None:
            ndim = 1 if ndim is None else ndim
            return LogicalStore.from_handle(
                self._handle.create_store(dtype._handle, <uint32_t> ndim)
            )

        if not is_shape_like(shape):
            raise ValueError(
                "shape must be a Shape object or an iterable"
            )

        cdef _Shape cpp_shape
        for extent in shape:
            try:
                cpp_shape.append_inplace(<size_t> extent)
            except OverflowError:
                raise ValueError("Extent must be a positive number")

        return LogicalStore.from_handle(
            self._handle.create_store(
                std_move(cpp_shape), dtype._handle, optimize_scalar
            )
        )

    def create_store_from_scalar(
        self,
        scalar: Scalar,
        shape: Optional[Union[Shape, Iterable[int]]] = None,
    ) -> LogicalStore:
        if shape is None:
            return LogicalStore.from_handle(
                self._handle.create_store(scalar._handle)
            )

        if not is_shape_like(shape):
            raise ValueError(
                "shape must be a Shape object or an iterable"
            )

        cdef _Shape cpp_shape
        for extent in shape:
            try:
                cpp_shape.append_inplace(<size_t> extent)
            except OverflowError:
                raise ValueError("Extent must be a positive number")

        return LogicalStore.from_handle(
            self._handle.create_store(scalar._handle, std_move(cpp_shape))
        )

    def create_store_from_bytes(
        self,
        Type dtype,
        shape: Optional[Union[Shape, Iterable[int]]],
        bytes data,
        bool share,
    ) -> LogicalStore:
        if not is_shape_like(shape):
            raise ValueError(
                "shape must be a Shape object or an iterable"
            )
        cdef _Shape cpp_shape = _Shape()
        for extent in shape:
            cpp_shape.append_inplace(<size_t> extent)

        cdef void* buf = <void*>(<char*> data)
        return LogicalStore.from_handle(
            self._handle.create_store(
                std_move(cpp_shape), dtype._handle, buf, share
            )
        )

    def create_store_from_buffer(
        self,
        Type dtype,
        shape: Optional[Union[Shape, Iterable[int]]],
        object data,
        bool share,
    ) -> LogicalStore:
        if not is_shape_like(shape):
            raise ValueError(
                "shape must be a Shape object or an iterable"
            )
        cdef _Shape cpp_shape = _Shape()
        for extent in shape:
            cpp_shape.append_inplace(<size_t> extent)

        cdef Py_buffer buffer
        PyObject_GetBuffer(
            data, &buffer, PyBUF_CONTIG if share else PyBUF_CONTIG_RO
        )

        return LogicalStore.from_handle(
            self._handle.create_store(
                std_move(cpp_shape), dtype._handle, buffer.buf, share
            )
        )

    def create_scalar(self, Type dtype, data: Any) -> Scalar:
        """
        Creates a future from a buffer holding a scalar value. The value is
        copied to the future.

        Parameters
        ----------
        data : buffer
            Buffer that holds a scalar value

        size : int
            Size of the value

        Returns
        -------
        Future
            A new future
        """
        return Scalar.create_from_buffer(data, dtype)

    def issue_execution_fence(self, bool block = False) -> None:
        self._handle.issue_execution_fence(block)

    def get_machine(self) -> Machine:
        return Machine.from_handle(self._handle.get_machine())

    @property
    def machine(self) -> Machine:
        return get_machine()

    def destroy(self) -> None:
        global _shutdown_manager
        _shutdown_manager.perform_callbacks()
        destroy()

    def push_machine(self, Machine machine) -> None:
        cdef _Machine cpp_machine = machine._handle
        self._handle.impl().machine_manager().push_machine(
            std_move(cpp_machine.impl().get()[0])
        )

    def pop_machine(self) -> None:
        self._handle.impl().machine_manager().pop_machine()

    def add_shutdown_callback(self, callback: ShutdownCallback) -> None:
        global _shutdown_manager
        _shutdown_manager.add_shutdown_callback(callback)


cdef Runtime initialize():
    cdef int32_t argc = len(sys.argv)
    cdef char** argv
    # TODO: Allocations we create here are leaked
    argv = <char**> malloc(argc * cython.sizeof(cython.p_char))
    for i, val in enumerate(sys.argv):
        arg = val.encode()
        argv[i] = <char*> malloc(len(arg) + 1)
        for j, v in enumerate(arg):
            argv[i][j] = <char> v
        argv[i][len(val)] = 0
    start(argc, argv)
    return Runtime.from_handle(_Runtime.get_runtime())


_runtime = initialize()


def get_legate_runtime() -> Runtime:
    """
    Returns the Legate runtime

    Returns
    -------
        Legate runtime object
    """
    global _runtime
    return _runtime


def get_machine() -> Machine:
    """
    Returns the machine of the current scope

    Returns
    -------
        Machine object
    """
    return _runtime.get_machine()


def caller_frameinfo() -> str:
    frame = inspect.currentframe()
    if frame is None:
        return "<unknown>"
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def track_provenance(
    nested: bool = False,
) -> Callable[[AnyCallable], AnyCallable]:
    """
    Decorator that adds provenance tracking to functions. Provenance of each
    operation issued within the wrapped function will be tracked automatically.

    Parameters
    ----------
    nested : bool
        If ``True``, each invocation to a wrapped function within another
        wrapped function updates the provenance string. Otherwise, the
        provenance is tracked only for the outermost wrapped function.

    Returns
    -------
    Decorator
        Function that takes a function and returns a one with provenance
        tracking

    See Also
    --------
    legate.core.runtime.Runtime.track_provenance
    """

    def decorator(func: AnyCallable) -> AnyCallable:
        if nested:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                provenance = caller_frameinfo()
                cdef _ProvenanceTracker* tracker = new _ProvenanceTracker(
                    provenance.encode()
                )
                result = func(*args, **kwargs)
                del tracker
                return result
        else:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                cdef _Runtime* runtime = (
                    <Runtime> get_legate_runtime()
                )._handle
                prov_mgt = runtime.impl().provenance_manager()
                had_prov = prov_mgt.has_provenance()
                if not had_prov:
                    prov_mgt.push_provenance(caller_frameinfo().encode())
                result = func(*args, **kwargs)
                if not had_prov:
                    prov_mgt.pop_provenance()
                return result

        return wrapper

    return decorator


def _cleanup_legate_runtime() -> None:
    get_legate_runtime().destroy()
    gc.collect()


add_cleanup_item(_cleanup_legate_runtime)
