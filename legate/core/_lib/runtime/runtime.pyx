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

from libc.stdint cimport int32_t, int64_t, uint32_t
from libc.stdlib cimport malloc as std_malloc
from libcpp cimport bool
from libcpp.utility cimport move as std_move

from ..._ext.cython_libcpp.string_view cimport (
    string_view_from_py as std_string_view_from_py,
)

import gc
import inspect
import sys
from typing import Any, Iterable

cimport cython

from legion_top import add_cleanup_item

from ..data.external_allocation cimport _ExternalAllocation, create_from_buffer
from ..data.logical_array cimport (
    LogicalArray,
    _LogicalArray,
    to_cpp_logical_array,
)
from ..data.logical_store cimport LogicalStore
from ..data.scalar cimport Scalar
from ..data.shape cimport Shape
from ..mapping.machine cimport Machine
from ..operation.task cimport AutoTask, ManualTask
from ..type.type_info cimport Type
from ..utilities.utils cimport (
    domain_from_iterables,
    is_iterable,
    uint64_tuple_from_iterable,
)
from .library cimport Library
from .tracker cimport _ProvenanceTracker

from ...utils import AnyCallable, ShutdownCallback


cdef class ShutdownCallbackManager:
    cdef list[ShutdownCallback] _shutdown_callbacks

    def __init__(self) -> None:
        self._shutdown_callbacks = []

    cdef void add_shutdown_callback(self, callback: ShutdownCallback):
        self._shutdown_callbacks.append(callback)

    cdef void perform_callbacks(self):
        for callback in self._shutdown_callbacks:
            callback()


cdef ShutdownCallbackManager _shutdown_manager = ShutdownCallbackManager()


create_legate_task_exception()
LegateTaskException = <object> _LegateTaskException


cdef void _reraise_legate_exception(
    tuple[type] exception_types, Exception e
) except *:
    cdef int index

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

    def __init__(self) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )

    cpdef Library find_library(self, str library_name):
        return Library.from_handle(
            self._handle.find_library(std_string_view_from_py(library_name))
        )

    @property
    def core_library(self) -> Library:
        return self.find_library("legate.core")

    cpdef AutoTask create_auto_task(self, Library library, int64_t task_id):
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

    cpdef ManualTask create_manual_task(
        self,
        Library library,
        int64_t task_id,
        object launch_shape,
        object lower_bounds = None,
    ):
        """
        Creates a manual task.

        When ``lower_bounds`` is None, the task's launch domain is ``[0,
        launch_shape)``. Otherwise, the launch domain is ``[lower_bounds,
        launch_shape)``.

        Parameters
        ----------
        library: Library
            Library to which the task id belongs

        task_id : int
            Task id. Scoped locally within the library; i.e., different
            libraries can use the same task id. There must be a task
            implementation corresponding to the task id.

        launch_shape : tuple
            Launch shape of the task

        lower_bounds : tuple, optional
            Optional lower bounds for the launch domain

        Returns
        -------
        ManualTask
            A new task
        """
        if not is_iterable(launch_shape):
            raise ValueError("Launch space must be iterable")

        if lower_bounds is not None and not is_iterable(lower_bounds):
            raise ValueError("Lower bounds must be iterable")

        if lower_bounds is None:
            return ManualTask.from_handle(
                self._handle.create_task(
                    library._handle,
                    task_id,
                    uint64_tuple_from_iterable(launch_shape),
                )
            )
        cdef int v
        return ManualTask.from_handle(
            self._handle.create_task(
                library._handle, task_id, domain_from_iterables(
                    lower_bounds,
                    tuple([v - 1 for v in launch_shape]),
                )
            )
        )

    cpdef void issue_copy(
        self,
        LogicalStore target,
        LogicalStore source,
        redop: int | None = None,
    ):
        if redop is None:
            self._handle.issue_copy(target._handle, source._handle)
        else:
            self._handle.issue_copy(
                target._handle, source._handle, <int32_t> redop
            )

    cpdef void issue_gather(
        self,
        LogicalStore target,
        LogicalStore source,
        LogicalStore source_indirect,
        redop: int | None = None,
    ):
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

    cpdef void issue_scatter(
        self,
        LogicalStore target,
        LogicalStore target_indirect,
        LogicalStore source,
        redop: int | None = None,
    ):
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

    cpdef void issue_scatter_gather(
        self,
        LogicalStore target,
        LogicalStore target_indirect,
        LogicalStore source,
        LogicalStore source_indirect,
        redop: int | None = None,
    ):
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

    cpdef void issue_fill(self, object array_or_store, object value):
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

    cpdef LogicalStore tree_reduce(
        self,
        Library library,
        int64_t task_id,
        LogicalStore store,
        int64_t radix = 4,
    ):
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

    cpdef void submit(self, object op):
        if isinstance(op, AutoTask):
            try:
                with nogil:
                    self._handle.submit((<AutoTask> op)._handle)
                return
            except LegateTaskException as e:
                _reraise_legate_exception(op.exception_types, e)
        elif isinstance(op, ManualTask):
            try:
                with nogil:
                    self._handle.submit((<ManualTask> op)._handle)
                return
            except LegateTaskException as e:
                _reraise_legate_exception(op.exception_types, e)

        raise RuntimeError(f"Unknown type of operation: {type(op)}")

    cpdef LogicalArray create_array(
        self,
        Type dtype,
        shape: Shape | Iterable[int] | None = None,
        bool nullable = False,
        bool optimize_scalar = False,
        ndim: int | None = None,
    ):
        if ndim is not None and shape is not None:
            raise ValueError("ndim cannot be used with shape")

        if ndim is None and shape is None:
            ndim = 1

        if shape is None:
            return LogicalArray.from_handle(
                self._handle.create_array(dtype._handle, ndim, nullable)
            )

        return LogicalArray.from_handle(
            self._handle.create_array(
                Shape.from_shape_like(shape),
                dtype._handle,
                nullable,
                optimize_scalar,
            )
        )

    cpdef LogicalArray create_array_like(self, LogicalArray array, Type dtype):
        return LogicalArray.from_handle(
            self._handle.create_array_like(
                array._handle, dtype._handle
            )
        )

    cpdef LogicalStore create_store(
        self,
        Type dtype,
        shape: Shape | Iterable[int] | None = None,
        bool optimize_scalar = False,
        ndim: int | None = None,
    ):
        if ndim is not None and shape is not None:
            raise ValueError("ndim cannot be used with shape")

        if shape is None:
            ndim = 1 if ndim is None else ndim
            return LogicalStore.from_handle(
                self._handle.create_store(dtype._handle, <uint32_t> ndim)
            )

        return LogicalStore.from_handle(
            self._handle.create_store(
                Shape.from_shape_like(shape), dtype._handle, optimize_scalar
            )
        )

    cpdef LogicalStore create_store_from_scalar(
        self,
        Scalar scalar,
        shape: Shape | Iterable[int] | None = None,
    ):
        if shape is None:
            return LogicalStore.from_handle(
                self._handle.create_store(scalar._handle)
            )

        return LogicalStore.from_handle(
            self._handle.create_store(
                scalar._handle, Shape.from_shape_like(shape)
            )
        )

    cpdef LogicalStore create_store_from_buffer(
        self,
        Type dtype,
        shape: Shape | Iterable[int],
        object data,
        bool read_only,
    ):
        """
        Creates a Legate store from a Python object implementing the Python
        buffer protocol.

        Parameters
        ----------
        dtype: Type
            Type of the store's elements

        shape : Shape or Iterable[int]
            Shape of the store

        data : object
            A Python object that implements the Python buffer protocol

        read_only : bool
            Whether the buffer of the passed object should be treated read-only
            or not. If ``False``, any changes made to the store will also be
            visible via the Python object.

        Returns
        -------
        LogicalStore
            Logical store attached to the buffer of the passed object

        Raises
        ------
        BufferError
            If the passed object does not implement the Python buffer protocol

        Notes
        -----
        It's the callers' responsibility to make sure that buffers passed to
        this function are never partially aliased; i.e., Python objects ``A``
        and ``B`` passed to this function must be backed by either the exact
        same buffer or two non-overlapping buffers. The code will exhibit
        undefined behavior in the presence of partial aliasing.
        """
        cdef _Shape cpp_shape = Shape.from_shape_like(shape)
        cdef _ExternalAllocation alloc = create_from_buffer(
            data, cpp_shape.volume() * dtype.size, read_only
        )
        return LogicalStore.from_handle(
            self._handle.create_store(
                std_move(cpp_shape), dtype._handle, std_move(alloc)
            )
        )

    cpdef void issue_execution_fence(self, bool block = False):
        with cython.nogil:
            # Must release the GIL in case we have in-flight python tasks,
            # since those can't run if we are stuck here holding the bag.
            self._handle.issue_execution_fence(block)

    cpdef Machine get_machine(self):
        return Machine.from_handle(self._handle.get_machine())

    @property
    def machine(self) -> Machine:
        return get_machine()

    cpdef void destroy(self):
        global _shutdown_manager
        _shutdown_manager.perform_callbacks()
        destroy()

    cpdef void push_machine(self, Machine machine):
        cdef _Machine cpp_machine = machine._handle
        self._handle.impl().machine_manager().push_machine(
            std_move(cpp_machine.impl().get()[0])
        )

    cpdef void pop_machine(self):
        self._handle.impl().machine_manager().pop_machine()

    cpdef void add_shutdown_callback(self, callback: ShutdownCallback):
        global _shutdown_manager
        _shutdown_manager.add_shutdown_callback(callback)


cdef Runtime initialize():
    cdef int32_t argc = len(sys.argv)
    # TODO(wonchanl): Allocations we create here are leaked
    cdef char** argv = <char**> std_malloc(argc * cython.sizeof(cython.p_char))
    cdef int i, j
    cdef str val
    cdef bytes arg
    for i, val in enumerate(sys.argv):
        arg = val.encode()
        argv[i] = <char*> std_malloc(len(arg) + 1)
        for j, v in enumerate(arg):
            argv[i][j] = <char> v
        argv[i][len(val)] = 0
    start(argc, argv)
    return Runtime.from_handle(_Runtime.get_runtime())


cdef Runtime _runtime = initialize()


cpdef Runtime get_legate_runtime():
    """
    Returns the Legate runtime

    Returns
    -------
        Legate runtime object
    """
    global _runtime
    return _runtime


cpdef Machine get_machine():
    """
    Returns the machine of the current scope

    Returns
    -------
        Machine object
    """
    return _runtime.get_machine()


cdef str caller_frameinfo():
    frame = inspect.currentframe()
    if frame is None:
        return "<unknown>"
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def track_provenance(
    bool nested = False
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
                cdef str provenance = caller_frameinfo()
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


cdef void _cleanup_legate_runtime():
    get_legate_runtime().destroy()
    gc.collect()


add_cleanup_item(_cleanup_legate_runtime)
