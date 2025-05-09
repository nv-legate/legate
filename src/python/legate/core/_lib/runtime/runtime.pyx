# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libcpp cimport bool
from libcpp.utility cimport move as std_move

import atexit

from ..._ext.cython_libcpp.string_view cimport (
    std_string_view,
    std_string_view_from_py,
)

import gc
import inspect
import json
import pickle
import sys
from collections.abc import Collection
from contextlib import contextmanager
from typing import Any, Iterator

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
from ..type.types cimport Type
from ..utilities.tuple cimport _tuple
from ..utilities.typedefs cimport _Domain, domain_from_iterables
from ..utilities.unconstructable cimport Unconstructable
from ..utilities.utils cimport (
    is_iterable,
    uint64_tuple_from_iterable,
)
from .library cimport Library, _Library

from ....settings import settings
from ...utils import AnyCallable, ShutdownCallback


class _LegateOutputStream:
    def __init__(self, stream: Any, node_id: int) -> None:
        self._stream = stream
        self._node_id = node_id

    def close(self) -> None:
        self._stream.close()

    def fileno(self) -> int:
        return self._stream.fileno()

    def flush(self) -> None:
        self._stream.flush()

    def write(self, string: str) -> None:
        if self._should_write:
            self._stream.write(string)

    def writelines(self, sequence: Iterable[str]) -> None:
        if self._should_write:
            self._stream.writelines(sequence)

    def isatty(self) -> bool:
        return self._stream.isatty()

    def set_parent(self, parent: Any) -> None:
        self._stream.set_parent(parent)

    @property
    def _should_write(self) -> bool:
        return self._node_id == 0 or is_running_in_task()


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
        r"""
        Find a `Library`.

        Parameters
        ----------
        library_name : str
            The name of the library.

        Returns
        -------
        Library
            The library.

        Raises
        ------
        ValueError
            If the library could not be found.
        """
        cdef std_string_view _library_name = std_string_view_from_py(
            library_name
        )
        cdef _Library handle

        try:
            with nogil:
                handle = self._handle.find_library(_library_name)
        except IndexError as ie:
            # C++ find_library() throws std::out_of_range, which Cython
            # converts to an IndexError. But ValueError is much more
            # appropriate (since what do indices have to do with not finding a
            # library name).
            raise ValueError(str(ie))

        return Library.from_handle(handle)

    # Same as `find_or_create_library()` except it also accepts a mapper
    # argument. The mapper isn't exposed to the user yet, so this remains a
    # cdef, not a cpdef.
    cdef tuple[Library, bool] find_or_create_library_mapper(
        self,
        str library_name,
        ResourceConfig config,
        std_unique_ptr[_Mapper] mapper,
        dict[VariantCode, VariantOptions] default_options,
    ):
        cdef std_string_view cpp_library_name = std_string_view_from_py(
            library_name
        )

        cdef std_map[VariantCode, _VariantOptions] cpp_default_options
        cdef VariantCode code
        cdef VariantOptions options

        for code, options in default_options.items():
            cpp_default_options[code] = options._handle

        cdef bool created
        cdef _Library handle

        with nogil:
            handle = self._handle.find_or_create_library(
                cpp_library_name,
                config._handle,
                std_move(mapper),
                cpp_default_options,
                &created
            )

        cdef Library lib = Library.from_handle(std_move(handle))

        return (lib, created)

    cpdef tuple[Library, bool] find_or_create_library(
        self,
        str library_name,
        ResourceConfig config = None,
        dict[VariantCode, VariantOptions] default_options = None
    ):
        r"""
        Search for an existing ``Library`` of a particular name, or create it
        if it wasn't found.

        Parameters
        ----------
        library_name : str
            The name of the library to find.
        config : ResourceConfig, optional
            The resource configuration to use to create the library if needed.
            Has no effect if the library is found.
        default_options : dict[VariantCode, VariantOptions], optional
            The default variant options to use to create the library if needed.
            Has no effect if the library is found.

        Returns
        -------
        Library
            The ``Library`` instance.
        bool
            ``True`` if the library was created, ``False`` otherwise.
        """
        if config is None:
            config = ResourceConfig()

        if default_options is None:
            default_options = {}

        return self.find_or_create_library_mapper(
            library_name=library_name,
            config=config,
            mapper=std_unique_ptr[_Mapper](),
            default_options=default_options,
        )

    cpdef Library create_library(self, str library_name):
        r"""
        Create a ``Library``.

        Parameters
        ----------
        library_name : str
            The name of the library to create.

        Returns
        -------
        Library
            The library.
        """
        cdef std_string_view lib_name = std_string_view_from_py(
            library_name
        )
        cdef _Library handle

        with nogil:
            handle = self._handle.create_library(lib_name)
        return Library.from_handle(std_move(handle))

    @property
    def core_library(self) -> Library:
        r"""
        Get the core library.

        :returns: The core library.
        :rtype: Library
        """
        return self.find_library("legate.core")

    cpdef AutoTask create_auto_task(
        self, Library library, _LocalTaskID task_id
    ):
        r"""
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
        r"""
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
        object redop = None,
    ):
        r"""
        Issue a copy between two stores.

        `source` and `target` must have the same shape.

        Parameters
        ----------
        target : LogicalStore
            The target.
        source : LogicalStore
            The source.
        redop : int (optional)
            The reduction operator to use. If none is given, no reductions take
            place. The stores type must support the operator.

        Raises
        ------
        ValueError
            If the store's type doesn't support `redop`.
        """
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
        object redop = None,
    ):
        r"""
        Issue a gather copy between stores.

        `source_indirect` and the `target` must have the same shape.

        Parameters
        ----------
        target : LogicalStore
            The target store.
        source : LogicalStore
            The source store.
        source_indirect : LogicalStore
            The source indirection store.
        redop : int (optional)
            ID of the reduction operator to use (optional). If none is given,
            no reductions take place. The store's type must support the
            operator.

        Raises
        ------
        ValueError
            If the store's type doesn't support `redop`.
        """
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
        object redop = None,
    ):
        r"""
        Issue a scatter copy between stores.

        `target_indirect` and the `source` must have the same shape.

        Parameters
        ----------
        target : LogicalStore
            The target store.
        target_indirect : LogicalStore
            The target indirection store.
        source : LogicalStore
            The source store.
        redop : int (optional)
            ID of the reduction operator to use (optional). If none is given,
            no reductions take place. The store's type must support the
            operator.

        Raises
        ------
        ValueError
            If the store's type doesn't support `redop`.
        """
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
        object redop = None,
    ):
        r"""
        Issue a scatter-gather copy between stores.

        `target_indirect` and the `source_indirect` must have the same shape.

        Parameters
        ----------
        target : LogicalStore
            The target store.
        target_indirect : LogicalStore
            The target indirection store.
        source : LogicalStore
            The source store.
        source_indirect : LogicalStore
            The source indirection store.
        redop : int (optional)
            ID of the reduction operator to use (optional). If none is given,
            no reductions take place. The store's type must support the
            operator.

        Raises
        ------
        ValueError
            If the store's type doesn't support `redop`.
        """
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
        r"""
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
        r"""
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
        r"""
        Submit a task for execution.

        Each submitted operation goes through multiple pipeline steps to
        eventually get scheduled for execution. It's not guaranteed that the
        submitted operation starts executing immediately.

        The exception to this rule is if the task indicates that it throws an
        exception. In this case, the scheduling pipeline is first flushed, then
        the task is executed. This routine does not return until the task has
        finished executing.

        If the task does *not* indicate that it throws an exception but throws
        one anyway, the runtime will catch and report the exception traceback
        then promptly abort the program.

        Parameters
        ----------
        op : AutoTask | ManualTask
            The task to submit.

        Raises
        ------
        Any
            If thrown, the exception raised by the task.
        TypeError
            If the operation is neither an `AutoTask` or `ManualTask`
        """
        if isinstance(op, AutoTask):
            try:
                with nogil:
                    self._handle.submit(std_move((<AutoTask> op)._handle))
                return
            except Exception as e:
                _maybe_reraise_legate_exception(e, op.exception_types)
        elif isinstance(op, ManualTask):
            try:
                with nogil:
                    self._handle.submit(std_move((<ManualTask> op)._handle))
                return
            except Exception as e:
                _maybe_reraise_legate_exception(e, op.exception_types)

        raise TypeError(f"Unknown type of operation: {type(op)}")

    cpdef LogicalArray create_array(
        self,
        Type dtype,
        shape: Shape | Collection[int] | None = None,
        bool nullable = False,
        bool optimize_scalar = False,
        object ndim = None,
    ):
        r"""
        Create a `LogicalArray`.

        If `shape` is `None`, the returned array is unbound, otherwise the
        array is bound.

        If not `None`, this call does not block on the value of `shape`.

        Parameters
        ----------
        dtype : Type
            The type of the array elements.
        shape : Shape | Collection[int] | None (optional)
            The shape of the array.
        nullable : bool (`False`)
            Whether the array is nullable.
        optimize_scalar : bool (`False`)
            Whether to optimize the array for scalar storage.
        ndim : int | None (optional)
            Number of dimensions.

        Returns
        -------
        LogicalArray
            The newly created array.

        Raises
        ------
        ValueError
            If both `ndim` and `shape` are simultaneously not `None`.
        """
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

    cpdef LogicalArray create_array_like(
        self, LogicalArray array, Type dtype = None
    ):
        r"""
        Create an array isomorphic to a given array.

        Parameters
        ----------
        array : LogicalArray
            The array to model the new array from.
        dtype : Type (optional)
            The type of the resulting array. If given, must be compatible with
            ``array``'s type. If not given, ``array``'s type is used.

        Returns
        -------
        LogicalArray
            The new array.
        """
        cdef _LogicalArray _handle

        if dtype is None:
            dtype = array.type

        with nogil:
            _handle = self._handle.create_array_like(
                array._handle, dtype._handle
            )
        return LogicalArray.from_handle(_handle)

    cpdef LogicalStore create_store(
        self,
        Type dtype,
        shape: Shape | Collection[int] | None = None,
        bool optimize_scalar = False,
        object ndim = None,
    ):
        r"""
        Create a `LogicalStore`.

        If `shape` is `None`, the created store is unbound, otherwise it is
        bound.

        If `shape` is not `None`, this call does not block on the shape.

        Parameters
        ----------
        dtype : Type
            The element type of the store.
        shape : Shape | Collection[int] | None (optional)
            The shape of the store.
        optimize_scalar : bool (`False`)
            Whether to optimize the store for scalar storage.
        ndim : int | None (optional, `1`)
            The number of dimensions for the store.

        Returns
        -------
        LogicalStore
            The newly created store.

        Raises
        ------
        ValueError
            If both `ndim` and `shape` are not `None`.
        """
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
        shape: Shape | Collection[int] | None = None,
    ):
        r"""
        Create a store from a `Scalar`.

        If `shape` is not `None`, its volume bust be `1`. The call does not
        block on the shape.

        Parameters
        ----------
        scalar : Scalar
            The scalar to create the store from.
        shape : Shape | Collection[int] | None (optional)
            The shape of the store.

        Returns
        -------
        LogicalStore
            The newly created store.
        """
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
        shape: Shape | Collection[int],
        object data,
        bool read_only,
    ):
        r"""
        Creates a Legate store from a Python object implementing the Python
        buffer protocol.

        Parameters
        ----------
        dtype: Type
            Type of the store's elements

        shape : Shape | Collection[int]
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

    cpdef void prefetch_bloated_instances(
        self,
        LogicalStore store,
        tuple low_offsets,
        tuple high_offsets,
        bool initialize = False,
    ):
        r"""
        Gives the runtime a hint that the store can benefit from bloated
        instances.

        The runtime currently does not look ahead in the task stream to
        recognize that a given set of tasks can benefit from the ahead-of-time
        creation of "bloated" instances encompassing multiple slices of a
        store. This means that the runtime will construct bloated instances
        incrementally and completely only when it sees all the slices,
        resulting in intermediate instances that (temporarily) increases the
        memory footprint. This function can be used to give the runtime a hint
        ahead of time about the bloated instances, which would be reused by the
        downstream tasks without going through the same incremental process.

        For example, let's say we have a 1-D store A of size 10 and we want to
        partition A across two GPUs. By default, A would be partitioned equally
        and each GPU gets an instance of size 5.  Suppose we now have a task
        that aligns two slices A[1:10] and A[:9]. The runtime would partition
        the slices such that the task running on the first GPU gets A[1:6] and
        A[:5], and the task running on the second GPU gets A[6:] and A[5:9].
        Since the original instance on the first GPU does not cover the element
        A[5] included in the first slice A[1:6], the mapper needs to create a
        new instance for A[:6] that encompasses both of the slices, leading to
        an extra copy.  In this case, if the code calls `prefetch(A, (0,),
        (1,))` to pre-alloate instances that contain one extra element on the
        right before it uses A, the extra copy can be avoided.

        A couple of notes about the API:

        - Unless `initialize` is `true`, the runtime assumes that the store has
          been initialized.  Passing an uninitialized store would lead to a
          runtime error.
        - If the store has pre-existing instances, the runtime may combine
          those with the bloated instances if such combination is deemed
          desirable.

        Parameters
        ----------
        store : LogicalStore
            Store to create bloated instances for
        low_offsets : tuple[int, ...]
            Offsets to bloat towards the negative direction
        high_offsets : tuple[int, ...]
            Offsets to bloat towards the positive direction
        initialize : bool, optional
            If `True`, the runtime will issue a fill on the store to initialize
            it. `False` by default.

        Notes
        -----
        This API is experimental.
        """

        cdef _LogicalStore handle = store._handle
        cdef _tuple[uint64_t] tup_lo = uint64_tuple_from_iterable(low_offsets)
        cdef _tuple[uint64_t] tup_hi = uint64_tuple_from_iterable(high_offsets)

        with nogil:
            self._handle.prefetch_bloated_instances(
                std_move(handle),
                std_move(tup_lo),
                std_move(tup_hi),
                initialize,
            )

    cpdef void issue_mapping_fence(self):
        r"""
        Issue a mapping fence.

        A mapping fence, when issued, blocks mapping of all downstream
        operations before those preceding the fence get mapped. An
        `issue_mapping_fence()` call returns immediately after the request is
        submitted to the runtime, and the fence asynchronously goes through the
        runtime analysis pipeline just like any other Legate operations. The
        call also flushes the scheduling window for batched execution.

        Mapping fences only affect how the operations are mapped and do not
        change their execution order, so they are semantically
        no-op. Nevertheless, they are sometimes useful when the user wants to
        control how the resource is consumed by independent tasks. Consider a
        program with two independent tasks `A` and `B`, both of which discard
        their stores right after their execution.  If the stores are too big to
        be allocated all at once, mapping A and B in parallel (which can happen
        because `A` and `B` are independent and thus nothing stops them from
        getting mapped concurrently) can lead to a failure. If a mapping fence
        exists between the two, the runtime serializes their mapping and can
        reclaim the memory space from stores that would be discarded after
        `A`'s execution to create allocations for `B`.
        """
        with nogil:
            self._handle.issue_mapping_fence()

    cpdef void issue_execution_fence(self, bool block = False):
        r"""
        Issue an execution fence.

        An execution fence is a join point in the task graph. All operations
        prior to a fence must finish before any of the subsequent operations
        start.

        All execution fences are mapping fences by definition; i.e., an
        execution fence not only prevents the downstream operations from being
        mapped ahead of itself but also precedes their execution.

        Parameters
        ----------
        block : bool (`False`)
            Whether to block control code on the fence. If `True`, this routine
            does not return until the scheduling pipeline has been fully
            flushed, and all tasks on it have finished executing.
        """
        with nogil:
            # Must release the GIL in case we have in-flight python tasks,
            # since those can't run if we are stuck here holding the bag.
            self._handle.issue_execution_fence(block)

    @property
    def node_count(self) -> uint32_t:
        r"""
        Get the total number of nodes.

        :returns: The total number of nodes.
        :rtype: int
        """
        return self._handle.node_count()

    @property
    def node_id(self) -> uint32_t:
        r"""
        Get the current rank.

        :returns: The current node rank.
        :rtype: int
        """
        return self._handle.node_id()

    cpdef Machine get_machine(self):
        r"""
        Get the current machine.

        Returns
        -------
        Machine
            The machine of the current scope.
        """
        return Machine.from_handle(self._handle.get_machine())

    @property
    def machine(self) -> Machine:
        r"""
        An alias for `get_machine()`.

        :returns: The current machine.
        :rtype: Machine
        """
        return get_machine()

    cpdef void finish(self):
        r"""
        Finish a Legate program.

        This routine:

        #. Performs all shutdown callbacks.
        #. Flushes the remaining scheduling pipeline.
        #. Issues a blocking execution fence.
        #. Tears down the runtime.

        It does not return until all steps above have completed.
        """
        global _shutdown_manager
        _shutdown_manager.perform_callbacks()
        with nogil:
            finish()

    cpdef void add_shutdown_callback(self, callback: ShutdownCallback):
        r"""
        Add a shutdown callback to be executed on Legate program finalization.

        Shutdown callbacks are only executed during normal program shutdown,
        and will not run if e.g. the program ends by exception or abort.

        Shutdown callbacks are executed in LIFO order. Callbacks may themselves
        register additional shutdown callbacks during their execution, though
        given the LIFO order, this means that the very next callback will be
        the function that was just registered.

        It is possible to register the same callback multiple times. No attempt
        is made to deduplicate these.

        Parameters
        ----------
        callback : ShutdownCallback
            The callback to register.
        """
        global _shutdown_manager
        _shutdown_manager.add_shutdown_callback(callback)

    cdef void start_profiling_range(self):
        self.start_profiling_range()

    cdef void stop_profiling_range(self, std_string_view provenance):
        self.stop_profiling_range(provenance)


cdef tuple[bool, bool] _set_realm_backtrace(str value):
    from os import environ

    cdef bool set_val

    # This could just be environ.setdefault(...) but we also need to detect
    # whether we set the value or not, which is not possible with setdefault().
    if "REALM_BACKTRACE" in environ:
        set_val = False
        value = environ["REALM_BACKTRACE"]
    else:
        set_val = True
        environ["REALM_BACKTRACE"] = value

    cdef int int_value

    try:
        int_value = int(value)
    except ValueError as ve:
        m = f"Invalid value for REALM_BACKTRACE: {value}"
        raise ValueError(m) from ve

    return set_val, <bool>int_value

cdef Runtime initialize():
    import faulthandler

    cdef bool set_val
    cdef bool value

    if faulthandler.is_enabled():
        set_val, value = _set_realm_backtrace("0")
        if value:
            m = (
                "REALM_BACKTRACE and the Python fault handler are mutually "
                "exclusive and cannot both be enabled."
            )
            raise RuntimeError(m)
    else:
        set_val, value = _set_realm_backtrace("1")

    start()
    runtime = Runtime.from_handle(_Runtime.get_runtime())

    if set_val:
        # This should just be lambda but Cython doesn't like that...
        #
        # if set_val:
        #    runtime.add_shutdown_callback(lambda: del environ["REALM_BACKTRACE"])
        #                                        ^
        # ------------------------------------------------------------
        #
        # /path/to/runtime.pyx:1213:46: Expected an identifier or literal
        def unset_realm_backtrace() -> None:
            from os import environ

            del environ["REALM_BACKTRACE"]

        runtime.add_shutdown_callback(unset_realm_backtrace)
    if settings.limit_stdout():
        sys.stdout = _LegateOutputStream(sys.stdout, runtime.node_id)
    return runtime


cdef Runtime _runtime = None

cdef void raise_pending_exception():
    cdef Runtime runtime = get_legate_runtime()
    try:
        with nogil:
            runtime._handle.raise_pending_exception()
    except Exception as e:
        _maybe_reraise_legate_exception(e)

cpdef Runtime get_legate_runtime():
    r"""
    Returns the Legate runtime.

    Returns
    -------
    Runtime
        Legate runtime object.
    """
    global _runtime
    if _runtime is None:
        _runtime = initialize()
    return _runtime


cpdef Machine get_machine():
    r"""
    Returns the machine of the current scope.

    Returns
    -------
    Machine
        Machine object.
    """
    return get_legate_runtime().get_machine()


cdef tuple _caller_frameinfo():
    frame = inspect.currentframe()
    if frame is None:
        return ()
    return (frame.f_code.co_filename, frame.f_lineno)


def _assemble_provenance(human: str, **machine: Any) -> str:
    return json.dumps([human, machine], indent=None)


cdef str _provenance_from_frameinfo(info: tuple[str, int]):
    if info == ():
        return _assemble_provenance("<unknown>", file="<unknown>")

    fname, lineno = info
    return _assemble_provenance(f"{fname}:{lineno}", file=fname, line=lineno)


def track_provenance(
    bool nested = False
) -> Callable[[AnyCallable], AnyCallable]:
    r"""
    Decorator that adds provenance tracking to functions. Provenance of each
    operation issued within the wrapped function will be tracked automatically.

    Parameters
    ----------
    nested : bool (`False`)
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
                cdef tuple info = _caller_frameinfo()
                cdef str provenance = _provenance_from_frameinfo(info)
                with Scope(provenance=provenance):
                    return func(*args, **kwargs)
        else:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if len(Scope.provenance()) > 0:
                    return func(*args, **kwargs)
                cdef tuple info = _caller_frameinfo()
                cdef str provenance = _provenance_from_frameinfo(info)
                with Scope(provenance=provenance):
                    return func(*args, **kwargs)

        return wrapper

    return decorator


cpdef bool is_running_in_task():
    r"""
    Determine whether the current control code is running inside a Legate task.

    Returns
    -------
    bool
        `True` if currently inside a Legate task, `False` otherwise.
    """
    return _is_running_in_task()


cdef void _cleanup_legate_runtime():
    global _runtime

    # Don't use get_legate_runtime() here since we don't want to inadvertently
    # (re)start the runtime
    if _runtime is None:
        return

    # Collect before so we ensure that any dangling user references are
    # released. We cannot guarantee that all of them are, but this should make
    # sure we maximize our chances.
    gc.collect()
    _runtime.finish()
    _runtime = None
    gc.collect()


@contextmanager
def ProfileRange(provenance: str) -> Iterator[None]:
    r"""
    Generate a sub-box in the profiler output.

    Parameters
    ----------
    provenance : str
        User-supplied provenance string to annotate the profiler output

    .. code-block:: python

        @task
        def foo():
            # do stuff
            with ProfileRange("range1"):
                # do stuff, to appear under a sub-box within foo's overall box
            # do stuff

    """
    cdef std_string_view _provenance = std_string_view_from_py(
        provenance
    )
    get_legate_runtime().start_profiling_range()
    try:
        yield
    finally:
        get_legate_runtime().stop_profiling_range(_provenance)


atexit.register(_cleanup_legate_runtime)
