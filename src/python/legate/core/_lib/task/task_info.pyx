# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

from libc.stdio cimport fflush as std_fflush, fprintf as std_fprintf, stderr
from libc.stdlib cimport abort as std_abort
from libcpp.string cimport string as std_string
from libcpp.unordered_map cimport unordered_map as std_unordered_map
from libcpp.utility cimport move as std_move

from ..._ext.cython_libcpp.string_view cimport (
    str_from_string_view,
    std_string_view
)
from ..runtime.library cimport Library, _Library
from ..runtime.runtime cimport get_legate_runtime
from ..utilities.typedefs cimport TaskFuncPtr, VariantCode, _GlobalTaskID
from ..utilities.unconstructable cimport Unconstructable
from .task_context cimport TaskContext, _TaskContext
from .variant_helper cimport task_wrapper_dyn_name
from .task_config cimport _TaskConfig, TaskConfig


cdef extern from *:
    # The vilest of hacks, known as the "cname-hack". Cython does not support
    # non-type template parameters, but it does allow you to define alternate
    # names for typedefs. To make a long story short, the typedef-ed names are
    # replaced *verbatim* with the alternate names in the final C/C++ code,
    # allowing for some advanced tomfoolery.
    #
    # So given foo[py_variant_t]() Cython believes py_variant_t to be a type
    # (literally, an alias to int), but replaces py_variant_t with
    # _py_variant. Therefore, the C/C++ compiler then sees foo<_py_variant>(),
    # i.e. a function template parameter, exactly as it expects.
    #
    # As a result, the alternate name must EXACTLY MATCH the "target" (in our
    # case, the function "_py_variant()" defined below). Everything else (the
    # name of the typedef, or the dummy type it aliases) is irrelevant.
    ctypedef int py_variant_t "_py_variant"
    ctypedef int LEGATE_CPU_VARIANT_T "legate::VariantCode::CPU"
    ctypedef int LEGATE_GPU_VARIANT_T "legate::VariantCode::GPU"
    ctypedef int LEGATE_OMP_VARIANT_T "legate::VariantCode::OMP"


cdef dict[_GlobalTaskID, dict[VariantCode, object]] \
    _gid_to_variant_callbacks = {}

# Note the alternate name, otherwise Cython mangles the resulting function name
# e.g. *__pyx_pf_7py_variant(...)
# Note also explicit "with gil", this is required as we are calling into Python
# code
cdef extern void _py_variant "_py_variant"(_TaskContext ctx) with gil:
    cdef TaskContext py_ctx = TaskContext.from_handle(&ctx)
    cdef _GlobalTaskID global_task_id = py_ctx.get_task_id()
    cdef VariantCode variant_kind = py_ctx.get_variant_kind()

    cdef std_string abort_message
    cdef dict variant_callbacks
    cdef object py_callback

    # Disable garbage collection while executing a python task, because some
    # Legate objects perform Legion runtime calls in their destructor (e.g.
    # LogicalStore can launch a Discard operation), which are not allowed in
    # leaf tasks. See https://github.com/nv-legate/legate.internal/issues/2261.
    # TODO(mpapadakis): We should make sure we never relinquish the GIL
    # and get context-switched out to the main thread (e.g. by calling a
    # runtime function). Otherwise that thread would be executing with
    # garbage collection disabled
    cdef bool gc_enabled_at_entry = gc.isenabled()
    if gc_enabled_at_entry:
        gc.disable()

    try:
        # Note: every one of these asserts must be inside this block, since
        # otherwise the errors from these assertions are never shown -- the
        # function silently returns.
        #
        # Why? Well, the TL;DR is that in order for a python exception to be
        # printed, someone has to call PyErr_Print() or PyErr_PrintEx(). This
        # function returns directly back into C++ land, so if this function
        # doesn't do it, nobody will.
        variant_callbacks = _gid_to_variant_callbacks.get(global_task_id)
        assert variant_callbacks is not None, (
            f"Task (global task id {global_task_id}) did not have any variant "
            "callbacks registered"
        )
        py_callback = variant_callbacks.get(variant_kind)
        assert py_callback is not None, (
            f"Task (global task id {global_task_id}) did not have a variant "
            f"registered for variant kind: {variant_kind}"
        )
        py_callback(py_ctx)
    except Exception as excn:
        # Cython does not know how to throw C++ exceptions, so we have to tell
        # the context to throw them later
        try:
            py_ctx.set_exception(excn)
        except Exception as excn2:
            try:
                abort_message = str(excn2).encode()
            except:  # noqa E722
                # couldn't even encode, something is truly wrong
                pass

            # We failed to set the exception for some reason. There is not
            # much that can really go wrong with that, so if that happened
            # we are well and truly hosed. Try one last ditch effort to
            # inform the user, and then abort
            std_fprintf(
                stderr,
                "Unhandled Python exception: '%s', aborting!\n",
                abort_message.c_str()
            )
            std_fflush(NULL)
            std_abort()
    finally:
        if gc_enabled_at_entry:
            gc.enable()

# Need an initializer function since I could not figure out how to initialize a
# std::unordered_map from a = {a : b, c : d} expression...
cdef std_unordered_map[VariantCode, TaskFuncPtr] _init_vmap():
    cdef std_unordered_map[VariantCode, TaskFuncPtr] result

    result.reserve(3)
    result[
        VariantCode.CPU
    ] = task_wrapper_dyn_name[py_variant_t, LEGATE_CPU_VARIANT_T]
    result[
        VariantCode.GPU
    ] = task_wrapper_dyn_name[py_variant_t, LEGATE_GPU_VARIANT_T]
    result[
        VariantCode.OMP
    ] = task_wrapper_dyn_name[py_variant_t, LEGATE_OMP_VARIANT_T]
    return result

cdef std_unordered_map[VariantCode, TaskFuncPtr] \
    _variant_to_callback = _init_vmap()


# Need to put this in a separate function because:
#
# 1. We need access to TaskInfo::AddVariantKey, and declaring nested C++
#    classes in Cython is a huge PITA. So it's easier to just write it out in
#    raw C++.
# 2. We need to control the name of the function. If we were to use a raw
#    Cython function, cython would mangle it. So we define the body in C++, and
#    tell Cython exactly what the name should be. Also, we want it in a
#    specific namespace because it makes the resulting friend decl in C++ more
#    obviously "Cython".
cdef extern from * nogil:
    """
    #include "legate.h"

    namespace legate::detail::cython {

    void cytaskinfo_add_variant(
      legate::TaskInfo *handle,
      const legate::Library &core_lib,
      legate::VariantCode variant_kind,
      legate::VariantImpl cy_entry,
      legate::Processor::TaskFuncPtr py_entry,
      const legate::TaskConfig& config)
    {
      handle->add_variant_(
        legate::TaskInfo::AddVariantKey{},
        core_lib,
        variant_kind,
        cy_entry,
        py_entry,
        config,
        /* decl_options */ nullptr
      );
    }

    } // namespace legate::detail::cython
    """
    void cytaskinfo_add_variant \
        "legate::detail::cython::cytaskinfo_add_variant" (
            _TaskInfo *,
            const _Library&,
            VariantCode,
            VariantImpl,
            TaskFuncPtr,
            const _TaskConfig&
        ) except+

cdef class TaskInfo(Unconstructable):
    @staticmethod
    cdef TaskInfo from_handle(_TaskInfo handle, _LocalTaskID local_task_id):
        cdef TaskInfo result = TaskInfo.__new__(TaskInfo)

        result._handle = std_move(handle)
        result._local_id = local_task_id
        result._registered_variants = {}
        return result

    cdef void validate_registered_py_variants(self):
        if not self._registered_variants:
            raise RuntimeError(
                f"Task (local id: {self.get_local_id()}) has no variants!"
            )

    cdef void register_global_variant_callbacks(
        self,
        _GlobalTaskID global_task_id
    ):
        global _gid_to_variant_callbacks

        if global_task_id in _gid_to_variant_callbacks:
            m = f"Already registered task (local id: {self.get_local_id()})"
            raise AssertionError(m)

        _gid_to_variant_callbacks[global_task_id] = self._registered_variants
        self._registered_variants = {}

    cdef _LocalTaskID get_local_id(self):
        return self._local_id

    def __repr__(self) -> str:
        r"""
        Return a human-readable string representation of the ``TaskInfo``.

        Returns
        -------
        str
            The string representation.
        """
        # must regular import here to get the python enum version
        from ..utilities.typedefs import VariantCode as py_VariantCode

        cdef list[str] descr = [
            vid.name
            for vid in py_VariantCode
            if self.has_variant(vid)
        ]
        cdef str variants = ", ".join(descr) if descr else "None"
        return f"TaskInfo(name: {self.name}, variants: {variants})"

    # The exact same thing as from_variants() (in fact, that function calls
    # this one), except that it also accepts a TaskConfig argument. The
    # reason we have two versions of the function is because TaskConfig does
    # not need to be exposed to the user.
    @staticmethod
    cdef TaskInfo from_variants_config(
        TaskConfig config,
        Library library,
        str name,
        list[tuple[VariantCode, object]] variants,
    ):
        cdef _LocalTaskID task_id = config.task_id

        if not variants:
            m = (
                "Cannot construct task info "
                f"(local id: {task_id}, name: {name})."
                " Variants must not be empty."
            )
            raise ValueError(m)

        cdef TaskInfo task_info

        task_info = TaskInfo.from_handle(
            _TaskInfo(name.encode()), task_id
        )

        cdef VariantCode variant_kind
        for variant_kind, variant_fn in variants:
            task_info.add_variant_config(
                config=config,
                library=library,
                variant_kind=variant_kind,
                fn=variant_fn
            )
        return task_info

    @classmethod
    def from_variants(
        cls,
        _LocalTaskID local_task_id,
        str name,
        list[tuple[VariantCode, object]] variants
    ) -> TaskInfo:
        r"""
        Construct a ``TaskInfo`` from a list of variants.

        Parameters
        ----------
        local_task_id : LocalTaskID
            The local task ID of the task.
        name : str
            The name of the task.
        variants : list[tuple[VariantCode, VariantFunction]]
            The variants to register.

        Returns
        -------
        TaskInfo
            The created ``TaskInfo`` object.
        """
        return TaskInfo.from_variants_config(
            config=TaskConfig(local_task_id),
            library=get_legate_runtime().core_library,
            name=name,
            variants=variants
        )

    @property
    def name(self) -> str:
        r"""
        Get the name of the task.

        :returns: The task name.
        :rtype: str

        :raises RuntimeError: If the task info object is in an invalid state.
        """
        cdef std_string_view sv

        with nogil:
            sv = self._handle.name()
        return str_from_string_view(sv)

    cpdef bool has_variant(self, VariantCode variant_id):
        r"""
        Get whether a `TaskInfo` object has a particular variant.

        Parameters
        ----------
        variant_id : VariantCode
            The variant kind to query.

        Returns
        -------
        bool
            ``True`` if the variant exists, ``False`` otherwise.

        Raises
        ------
        RuntimeError
            If the task info object is in an invalid state.
        """
        cdef bool ret

        with nogil:
            ret = self._handle.find_variant(variant_id).has_value()
        return ret

    cdef void add_variant_config(
        self,
        TaskConfig config,
        Library library,
        VariantCode variant_kind,
        object fn
    ):
        if not callable(fn):
            raise TypeError(
                f"Variant function ({fn}) for variant kind {variant_kind} is "
                "not callable"
            )

        # do this check before we call into C++ since we cannot undo the
        # registration
        cdef TaskFuncPtr callback
        try:
            callback = _variant_to_callback[variant_kind]
        except KeyError as ke:
            raise ValueError(f"Unknown variant '{variant_kind}'") from ke

        if variant_kind in self._registered_variants:
            raise RuntimeError(
                "Already added callback "
                f"({self._registered_variants[variant_kind]}) "
                f"for {variant_kind} variant "
                f"(local id: {self.get_local_id()})"
            )

        with nogil:
            cytaskinfo_add_variant(
                &self._handle,
                library._handle,
                variant_kind,
                _py_variant,
                callback,
                config._handle
            )
        # do this last to preserve strong exception guarantee
        self._registered_variants[variant_kind] = fn

    cpdef void add_variant(self, VariantCode variant_kind, object fn):
        r"""
        Register a variant to a `TaskInfo`.

        Parameters
        ----------
        variant_kind : VariantCode
            The variant kind to add.
        fn : VariantFunction
            The variant to add.

        Raises
        ------
        RuntimeError
            If the task info object is in an invalid state.
        TypeError
            If ``fn`` is not callable.
        ValueError
            If ``variant_kind`` is an unknown variant kind.
        RuntimeError
            If the task info object has already registered a variant for
            ``variant_kind``.
        """
        self.add_variant_config(
            config=TaskConfig(self._local_id),
            library=get_legate_runtime().core_library,
            variant_kind=variant_kind,
            fn=fn
        )
