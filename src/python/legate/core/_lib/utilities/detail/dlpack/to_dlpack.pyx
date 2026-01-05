# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cimport cython
from cpython cimport (
    PyCapsule_IsValid,
    PyCapsule_New,
    PyCapsule_GetPointer
)

from libc.stdint cimport uintptr_t
from libcpp.utility cimport move as std_move
from libcpp.optional cimport nullopt as std_nullopt

from .dlpack cimport VERSIONED_CAPSULE_NAME_USED, VERSIONED_CAPSULE_NAME


cdef void dlpack_capsule_deleter(object capsule) noexcept:
    if PyCapsule_IsValid(capsule, VERSIONED_CAPSULE_NAME_USED):
        return

    cdef _DLManagedTensorVersioned *managed = <_DLManagedTensorVersioned *>(
        PyCapsule_GetPointer(capsule, VERSIONED_CAPSULE_NAME)
    )

    if managed == NULL:
        m = (
            "DLPack PyCapsule contained a NULL pointer "   # pragma: no cover
            "for the tensor."
        )
        raise ValueError(m)  # pragma: no cover

    if managed.deleter:
        managed.deleter(managed)


# The special cyton.cpp_locals directive is needed because Cython does not
# understand how to initialize unique_ptr's with custom deleters:
#
# /path/to/_lib/utilities/detail/dlpack/to_dlpack.cxx:4160:87:
# error: no matching constructor for initialization of 'std::unique_ptr<
# ::DLManagedTensorVersioned, void (*)(::DLManagedTensorVersioned *)>'
# 4160 | std::unique_ptr<...>  __pyx_v_dlm_tensor;
#      |                       ^
#
# We'd need to be able to do something like:
#
# std::unique_ptr ptr{pointer, deleter};
#
# But since Cython splits up every variable declaration from it initializer
# (i.e. the following):
#
# cdef Foo foo = make_foo()
#
# Is actually transpiled by Cython as
#
# Foo foo;
# ...
# foo = make_foo();
#
# This isn't possible.
@cython.cpp_locals(True)
cpdef object to_dlpack(
    PhysicalStore store,
    object stream = None,
    object max_version = None,
    object dl_device = None,
    object copy = None
):
    cdef std_optional[CUstream] cpp_stream = std_nullopt
    cdef uintptr_t cu_stream = 0

    if stream is not None:
        if stream == -1:
            # Ignore -1 (request to not sync), see below for rationale
            stream = 0
        cu_stream = <uintptr_t>stream
        # From DLPack docs, the values for stream:
        #
        # - None: producer must assume the legacy default stream (default).
        # - 1: the legacy default stream.
        # - 2: the per-thread default stream.
        # - >2: stream number represented as a Python integer.
        # - 0 is disallowed due to its ambiguity: 0 could mean either None, 1,
        #   or 2.
        #
        # We don't unconditionally set it to zero, because if there is a stream
        # set, then the C++ code will attempt to sync with it, even if it
        # doesn't have CUDA support (and therefore fail).
        #
        # The spec also says:
        #
        # > If stream is -1, the value may be used by the consumer to signal
        #   "producer must not perform any synchronization".
        #
        # But we choose to ignore it, (which is within our rights), because it
        # makes no sense. If the user doesn't give us a stream to signal the
        # completion of the transfer, and then asks us to not perform any
        # synchronization, then how in the world does the user know when it's
        # safe to access the data?
        if cu_stream <= 2:
            cu_stream = 0   # legacy default stream
        cpp_stream = <CUstream>cu_stream

    cdef std_optional[_DLPackVersion] cpp_max_version = std_nullopt

    if max_version is not None:
        cpp_max_version = _DLPackVersion(max_version[0], max_version[1])

    cdef std_optional[_DLDevice] cpp_device = std_nullopt

    if dl_device is not None:
        cpp_device = _DLDevice(dl_device[0], dl_device[1])

    cdef std_optional[bool] cpp_copy = std_nullopt

    if copy is not None:
        # Coerce to bool instead of hard <cast> to better handle truthy values
        cpp_copy = True if copy else False

    cdef std_unique_ptr[
        _DLManagedTensorVersioned, void(*)(_DLManagedTensorVersioned*)
    ] dlm_tensor

    try:
        with nogil:
            dlm_tensor = _to_dlpack(
                store=store._handle,
                copy=std_move(cpp_copy),
                stream=std_move(cpp_stream),
                max_version=std_move(cpp_max_version),
                device=std_move(cpp_device)
            )
    except Exception as e:
        raise BufferError from e

    return PyCapsule_New(
        dlm_tensor.release(),
        VERSIONED_CAPSULE_NAME,
        dlpack_capsule_deleter
    )
