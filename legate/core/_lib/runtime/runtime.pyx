# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr as std_unique_ptr
from libcpp.utility cimport move as std_move

import atexit

from ..._ext.cython_libcpp.string_view cimport (
    string_view as std_string_view,
    string_view_from_py as std_string_view_from_py,
)

import gc
import inspect
import pickle
import sys
from collections.abc import Iterable
from typing import Any


from ..data.external_allocation cimport _ExternalAllocation, create_from_buffer
from ..data.logical_array cimport (
    LogicalArray,
    _LogicalArray,
    to_cpp_logical_array,
)
from ..data.logical_store cimport LogicalStore, _LogicalStore
from ..data.scalar cimport Scalar
from ..data.shape cimport Shape, _Shape
from ..mapping.machine cimport Machine
from ..operation.task cimport AutoTask, ManualTask, _AutoTask, _ManualTask
from ..runtime.scope cimport Scope
from ..type.type_info cimport Type
from ..utilities.tuple cimport _tuple
from ..utilities.typedefs cimport _Domain
from ..utilities.unconstructable cimport Unconstructable
from ..utilities.utils cimport (
    domain_from_iterables,
    is_iterable,
    uint64_tuple_from_iterable,
)
from .library cimport Library, _Library

from ...utils import AnyCallable, ShutdownCallback


cdef class ShutdownCallbackManager:
    cdef list[ShutdownCallback] _shutdown_callbacks

    def __init__(self) -> None:
        self._shutdown_callbacks = []

    cdef void add_shutdown_callback(self, callback: ShutdownCallback):
        self._shutdown_callbacks.append(callback)

    cdef void perform_callbacks(self):
        # This while form (instead of the usual for x in list) is
        # deliberate. We want to iterate in LIFO order, while also allowing
        # each shutdown callback to potentially register other shutdown
        # callbacks.
        while self._shutdown_callbacks:
            callback = self._shutdown_callbacks.pop()
            callback()

cdef ShutdownCallbackManager _shutdown_manager = ShutdownCallbackManager()


create_legate_task_exceptions()
LegateTaskException = <object> _LegateTaskException
LegatePyTaskException = <object> _LegatePyTaskException


cdef void _maybe_reraise_legate_exception(
    Exception e, exception_types : tuple[type] | None = None,
) except *:
    cdef str message
    cdef bytes pkl_bytes
    cdef Exception exn_obj
    cdef int index

    if isinstance(e, LegatePyTaskException):
        (pkl_bytes,) = e.args
        exn_obj = pickle.loads(pkl_bytes)
        raise exn_obj
    if isinstance(e, LegateTaskException):
        (message, index) = e.args
        try:
            exn_type = (
                RuntimeError
                if exception_types is None
                else exception_types[index]
            )
        except IndexError:
            raise RuntimeError(
                f"Invalid exception index ({index}) while mapping task "
                f"exception: \"{message}\""
            )
        raise exn_type(message)
    raise

cdef class Runtime(Unconstructable):
    @staticmethod
    cdef Runtime from_handle(_Runtime* handle):
        cdef Runtime result = Runtime.__new__(Runtime)
        result._handle = handle
        return result

    cpdef Library find_library(self, str library_name):
        cdef std_string_view _library_name = std_string_view_from_py(
            library_name
        )
        cdef _Library handle
        with nogil:
            handle = self._handle.find_library(_library_name)
        return Library.from_handle(handle)

    @property
    def core_library(self) -> Library:
        return self.find_library("legate.core")

    cpdef AutoTask create_auto_task(
        self, Library library, _LocalTaskID task_id
    ):
        """
        Creates an auto task.

        Parameters
        ----------
        library: Library
            Library to which the task id belongs

        task_id : LocalTaskID
            Task id. Scoped locally within the library; i.e., different
            libraries can use the same task id. There must be a task
            implementation corresponding to the task id.

        Returns
        -------
        AutoTask
            A new automatically parallelized task
        """
        cdef _AutoTask handle
        with nogil:
            handle = self._handle.create_task(library._handle, task_id)
        return AutoTask.from_handle(handle)

    cpdef ManualTask create_manual_task(
        self,
        Library library,
        _LocalTaskID task_id,
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

        task_id : LocalTaskID
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

        cdef _ManualTask handle
        cdef int v
        cdef _tuple[uint64_t] _launch_shape
        cdef _Domain _launch_domain

        if lower_bounds is None:
            _launch_shape = uint64_tuple_from_iterable(launch_shape)

            with nogil:
                handle = self._handle.create_task(
                    library._handle,
                    task_id,
                    _launch_shape,
                )
        else:
            _launch_domain = domain_from_iterables(
                lower_bounds, tuple([v - 1 for v in launch_shape]),
            )

            with nogil:
                handle = self._handle.create_task(
                    library._handle, task_id, _launch_domain
                )
        return ManualTask.from_handle(handle)

    cpdef void issue_copy(
        self,
        LogicalStore target,
        LogicalStore source,
        redop: int | None = None,
    ):
        cdef int32_t _redop

        if redop is None:
            with nogil:
                self._handle.issue_copy(target._handle, source._handle)
        else:
            _redop = <int32_t> redop

            with nogil:
                self._handle.issue_copy(target._handle, source._handle, _redop)

    cpdef void issue_gather(
        self,
        LogicalStore target,
        LogicalStore source,
        LogicalStore source_indirect,
        redop: int | None = None,
    ):
        cdef int32_t _redop

        if redop is None:
            with nogil:
                self._handle.issue_gather(
                    target._handle,
                    source._handle,
                    source_indirect._handle,
                )
        else:
            _redop = <int32_t> redop

            with nogil:
                self._handle.issue_gather(
                    target._handle,
                    source._handle,
                    source_indirect._handle,
                    _redop,
                )

    cpdef void issue_scatter(
        self,
        LogicalStore target,
        LogicalStore target_indirect,
        LogicalStore source,
        redop: int | None = None,
    ):
        cdef int32_t _redop

        if redop is None:
            with nogil:
                self._handle.issue_scatter(
                    target._handle,
                    target_indirect._handle,
                    source._handle,
                )
        else:
            _redop = <int32_t> redop

            with nogil:
                self._handle.issue_scatter(
                    target._handle,
                    target_indirect._handle,
                    source._handle,
                    _redop,
                )

    cpdef void issue_scatter_gather(
        self,
        LogicalStore target,
        LogicalStore target_indirect,
        LogicalStore source,
        LogicalStore source_indirect,
        redop: int | None = None,
    ):
        cdef int32_t _redop

        if redop is None:
            with nogil:
                self._handle.issue_scatter_gather(
                    target._handle,
                    target_indirect._handle,
                    source._handle,
                    source_indirect._handle,
                )
        else:
            _redop = <int32_t> redop

            with nogil:
                self._handle.issue_scatter_gather(
                    target._handle,
                    target_indirect._handle,
                    source._handle,
                    source_indirect._handle,
                    _redop,
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
        cdef Scalar fill_value

        if isinstance(value, LogicalStore):
            with nogil:
                self._handle.issue_fill(arr, (<LogicalStore> value)._handle)
        elif isinstance(value, Scalar):
            with nogil:
                self._handle.issue_fill(arr, (<Scalar> value)._handle)
        elif value is None:
            fill_value = Scalar.null()
            with nogil:
                self._handle.issue_fill(arr, fill_value._handle)
        else:
            fill_value = Scalar(value, Type.from_handle(arr.type()))
            with nogil:
                self._handle.issue_fill(arr, fill_value._handle)

    cpdef LogicalStore tree_reduce(
        self,
        Library library,
        _LocalTaskID task_id,
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

        task_id : LocalTaskID
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
        cdef _LogicalStore _handle
        with nogil:
            _handle = self._handle.tree_reduce(
                library._handle, task_id, store._handle, radix
            )
        return LogicalStore.from_handle(_handle)

    cpdef void submit(self, object op):
        if isinstance(op, AutoTask):
            try:
                with nogil:
                    self._handle.submit((<AutoTask> op)._handle)
                return
            except Exception as e:
                _maybe_reraise_legate_exception(e, op.exception_types)
        elif isinstance(op, ManualTask):
            try:
                with nogil:
                    self._handle.submit((<ManualTask> op)._handle)
                return
            except Exception as e:
                _maybe_reraise_legate_exception(e, op.exception_types)

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

        cdef _LogicalArray _handle
        cdef _Shape _shape
        cdef uint32_t _ndim

        if shape is None:
            _ndim = ndim

            with nogil:
                _handle = self._handle.create_array(
                    dtype._handle, _ndim, nullable
                )
        else:
            _shape = Shape.from_shape_like(shape)
            with nogil:
                _handle = self._handle.create_array(
                    _shape,
                    dtype._handle,
                    nullable,
                    optimize_scalar,
                )

        return LogicalArray.from_handle(_handle)

    cpdef LogicalArray create_array_like(self, LogicalArray array, Type dtype):
        cdef _LogicalArray _handle

        with nogil:
            _handle = self._handle.create_array_like(
                array._handle, dtype._handle
            )
        return LogicalArray.from_handle(_handle)

    cpdef LogicalStore create_store(
        self,
        Type dtype,
        shape: Shape | Iterable[int] | None = None,
        bool optimize_scalar = False,
        ndim: int | None = None,
    ):
        if ndim is not None and shape is not None:
            raise ValueError("ndim cannot be used with shape")

        cdef uint32_t _ndim
        cdef _LogicalStore _handle
        cdef _Shape _shape

        if shape is None:
            _ndim = 1 if ndim is None else ndim
            with nogil:
                _handle = self._handle.create_store(dtype._handle, _ndim)
        else:
            _shape = Shape.from_shape_like(shape)
            with nogil:
                _handle = self._handle.create_store(
                    _shape, dtype._handle, optimize_scalar
                )

        return LogicalStore.from_handle(_handle)

    cpdef LogicalStore create_store_from_scalar(
        self,
        Scalar scalar,
        shape: Shape | Iterable[int] | None = None,
    ):
        cdef _LogicalStore _handle
        cdef _Shape _shape

        if shape is None:
            with nogil:
                _handle = self._handle.create_store(scalar._handle)
        else:
            _shape = Shape.from_shape_like(shape)

            with nogil:
                _handle = self._handle.create_store(
                    scalar._handle, _shape
                )

        return LogicalStore.from_handle(_handle)

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
        cdef _ExternalAllocation alloc
        try:
            alloc = create_from_buffer(
                data, cpp_shape.volume() * dtype.size, read_only
            )
        except BufferError as exn:
            raise ValueError(
                f"Passed buffer is too small for a store of shape {shape} and "
                f"type {dtype}"
            ) from exn
        cdef _LogicalStore _handle

        with nogil:
            _handle = self._handle.create_store(
                std_move(cpp_shape), dtype._handle, std_move(alloc)
            )
        return LogicalStore.from_handle(_handle)

    cpdef void issue_mapping_fence(self):
        with nogil:
            self._handle.issue_mapping_fence()

    cpdef void issue_execution_fence(self, bool block = False):
        with nogil:
            # Must release the GIL in case we have in-flight python tasks,
            # since those can't run if we are stuck here holding the bag.
            self._handle.issue_execution_fence(block)

    @property
    def node_count(self) -> uint32_t:
        return self._handle.node_count()

    @property
    def node_id(self) -> uint32_t:
        return self._handle.node_id()

    cpdef Machine get_machine(self):
        return Machine.from_handle(self._handle.get_machine())

    @property
    def machine(self) -> Machine:
        return get_machine()

    cpdef void finish(self):
        global _shutdown_manager
        _shutdown_manager.perform_callbacks()
        with nogil:
            finish()

    cpdef void add_shutdown_callback(self, callback: ShutdownCallback):
        global _shutdown_manager
        _shutdown_manager.add_shutdown_callback(callback)


cdef extern from *:
    """
    #include <new>
    #include <memory>
    #include <cstddef>

    namespace {

    template <typename T>
    std::unique_ptr<T[]> make_unique_array(std::size_t n)
    {
      return std::make_unique<T[]>(n);
    }

    } // namespace
    """
    # We need this helper because Cython does not support the array-new syntax,
    # i.e. new T[], which we need below in converting argv to char pointers.
    std_unique_ptr[T[]] make_unique_array[T](size_t)
    # Once again, working around the deficiencies of Cython. When you
    #
    # cdef SomeType[char *] v = a_template_fn[char *]()
    #                                        ~~~~~~~
    #              /----------------------------|
    #            -----
    # The second char * is not allowed, because Cython does not allow spaces in
    # the type names within template functions? Or maybe it's a bug in the
    # Cython compiler. But either way, the make_unique_array() call below does
    # not compile without this typedef
    ctypedef char *char_p "char *"

cdef Runtime initialize():
    cdef list sys_argv = sys.argv
    cdef int32_t argc = len(sys_argv)
    cdef std_unique_ptr[char *[]] argv = make_unique_array[char_p](argc + 1)
    cdef char **argv_ptr = <char **>argv.get()

    cdef str arg
    cdef list argv_bytes = [arg.encode() for arg in sys_argv]

    cdef int i
    cdef bytes arg_bytes

    for i, arg_bytes in enumerate(argv_bytes):
        argv_ptr[i] = arg_bytes

    cdef int32_t ret = start(argc, argv_ptr)

    if ret:
        raise RuntimeError(
            f"Failed to initialize legate runtime, return code: {ret}"
        )

    return Runtime.from_handle(_Runtime.get_runtime())


cdef Runtime _runtime = initialize()

cdef void raise_pending_exception():
    try:
        with nogil:
            _Runtime.get_runtime().raise_pending_exception()
    except Exception as e:
        _maybe_reraise_legate_exception(e)

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
    return get_legate_runtime().get_machine()


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
                with Scope(provenance=caller_frameinfo()):
                    return func(*args, **kwargs)
        else:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if len(Scope.provenance()) > 0:
                    return func(*args, **kwargs)
                with Scope(provenance=caller_frameinfo()):
                    return func(*args, **kwargs)

        return wrapper

    return decorator


cpdef bool is_running_in_task():
    return _is_running_in_task()


cdef void _cleanup_legate_runtime():
    get_legate_runtime().finish()
    gc.collect()


atexit.register(_cleanup_legate_runtime)
