# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libc.stdint cimport int64_t, uint32_t
from libc.stdio cimport fflush as std_fflush, fprintf as std_fprintf, stderr
from libc.stdlib cimport abort as std_abort
from libcpp.string cimport string as std_string
from libcpp.unordered_map cimport unordered_map as std_unordered_map

from ..legate_c cimport legate_core_variant_t
from ..utilities.typedefs cimport RealmCallbackFn
from .task_context cimport TaskContext, _TaskContext
from .variant_helper cimport task_wrapper_dyn_name


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
    ctypedef int LEGATE_CPU_VARIANT_T "LEGATE_CPU_VARIANT"
    ctypedef int LEGATE_GPU_VARIANT_T "LEGATE_GPU_VARIANT"
    ctypedef int LEGATE_OMP_VARIANT_T "LEGATE_OMP_VARIANT"


cdef dict[int64_t, dict[legate_core_variant_t, object]] \
    _gid_to_variant_callbacks = {}

# Note the alternate name, otherwise Cython mangles the resulting function name
# e.g. *__pyx_pf_7py_variant(...)
# Note also explicit "with gil", this is required as we are calling into Python
# code
cdef extern void _py_variant "_py_variant"(_TaskContext ctx) with gil:
    cdef TaskContext py_ctx = TaskContext.from_handle(&ctx)
    cdef int64_t global_task_id = py_ctx.get_task_id()
    cdef legate_core_variant_t variant_kind = py_ctx.get_variant_kind()

    cdef std_string abort_message
    cdef dict variant_callbacks
    cdef object py_callback
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
        assert variant_callbacks is not None
        py_callback = variant_callbacks.get(variant_kind)
        assert py_callback is not None
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

            # We failed to set the exception for some reason. There is not much
            # that can really go wrong with that, so if that happened we are
            # well and truly hosed. Try one last ditch effort to inform the
            # user, and then abort
            std_fprintf(
                stderr,
                "Unhandled Python exception: '%s', aborting!\n",
                abort_message.c_str()
            )
            std_fflush(NULL)
            std_abort()
    return

# Need an initializer function since I could not figure out how to initialize a
# std::unordered_map from a = {a : b, c : d} expression...
cdef std_unordered_map[legate_core_variant_t, RealmCallbackFn] _init_vmap():
    cdef std_unordered_map[legate_core_variant_t, RealmCallbackFn] result

    result[
        legate_core_variant_t._LEGATE_CPU_VARIANT
    ] = task_wrapper_dyn_name[py_variant_t, LEGATE_CPU_VARIANT_T]
    result[
        legate_core_variant_t._LEGATE_GPU_VARIANT
    ] = task_wrapper_dyn_name[py_variant_t, LEGATE_GPU_VARIANT_T]
    result[
        legate_core_variant_t._LEGATE_OMP_VARIANT
    ] = task_wrapper_dyn_name[py_variant_t, LEGATE_OMP_VARIANT_T]
    return result

cdef std_unordered_map[legate_core_variant_t, RealmCallbackFn] \
    _variant_to_callback = _init_vmap()

cdef class TaskInfo:
    @staticmethod
    cdef TaskInfo from_handle(
        _TaskInfo* p_handle, int64_t local_task_id
    ):
        cdef TaskInfo result = TaskInfo.__new__(TaskInfo)
        result._handle = p_handle
        result._local_id = local_task_id
        result._registered_variants = {}
        return result

    def __init__(self) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )

    cdef _TaskInfo *release(self) except NULL:
        if not self.valid:
            raise RuntimeError(
                "Cannot release the internal TaskInfo pointer because it is "
                "NULL"
            )

        cdef _TaskInfo *ptr = self._handle
        self._handle = NULL
        return ptr

    cdef void validate_registered_py_variants(self):
        if not self._registered_variants:
            raise RuntimeError(
                f"Task (local id: {self.get_local_id()}) has no variants!"
            )

    cdef void register_global_variant_callbacks(self, uint32_t global_task_id):
        assert global_task_id not in _gid_to_variant_callbacks, \
            f"Already registered task (local id: {self.get_local_id()})"
        _gid_to_variant_callbacks[global_task_id] = self._registered_variants
        self._registered_variants = {}
        return

    cdef int64_t get_local_id(self):
        return self._local_id

    def __dealloc__(self) -> None:
        if self.valid:
            del self._handle
        return

    def __repr__(self) -> str:
        # must regular import here to get the python enum version
        from ..legate_c import (
            legate_core_variant_t as py_legate_core_variant_t,
        )

        cdef list[str] descr = [
            vid.name
            for vid in py_legate_core_variant_t
            if self.has_variant(vid)
        ]
        cdef str variants = ", ".join(descr) if descr else "None"
        return f"TaskInfo(name: {self.name}, variants: {variants})"

    @classmethod
    def from_variants(
        cls,
        int64_t local_task_id,
        str name,
        list[tuple[legate_core_variant_t, object]] variants
    ) -> TaskInfo:
        if not variants:
            raise ValueError(
                "Cannot construct task info "
                f"(local id: {local_task_id}, name: {name})."
                " Variants must not be empty."
            )

        cdef _TaskInfo *tptr = new _TaskInfo(name.encode())
        cdef TaskInfo task_info
        try:
            task_info = TaskInfo.from_handle(tptr, local_task_id)
        except:  # noqa E722
            del tptr  # strong exception guarantee
            raise

        for variant_kind, variant_fn in variants:
            task_info.add_variant(variant_kind, variant_fn)
        return task_info

    @property
    def valid(self) -> bool:
        return self._handle != NULL

    @property
    def name(self) -> str:
        assert self.valid
        return self._handle.name().decode()

    cpdef bool has_variant(self, int variant_id):
        assert self.valid
        return self._handle.has_variant(<legate_core_variant_t>variant_id)

    cpdef void add_variant(
        self, legate_core_variant_t variant_kind, object fn
    ):
        assert self.valid
        assert callable(fn)

        # do this check before we call into C++ since we cannot undo the
        # registration
        cdef RealmCallbackFn callback
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

        self._handle.add_variant(variant_kind, _py_variant, callback)
        # do this last to preserve strong exception guarantee
        self._registered_variants[variant_kind] = fn
        return
