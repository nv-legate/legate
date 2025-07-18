# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.utility cimport move as std_move
from libc.stdint cimport uintptr_t

from cpython cimport PyCapsule_IsValid, PyCapsule_GetPointer, PyCapsule_SetName

from ....data.logical_store cimport LogicalStore

from .dlpack cimport (
    DLDeviceType,
    VERSIONED_CAPSULE_NAME,
    VERSIONED_CAPSULE_NAME_USED,
    CAPSULE_NAME,
    CAPSULE_NAME_USED,
    DLPACK_MAJOR_VERSION,
    DLPACK_MINOR_VERSION
)

cdef extern from * nogil:
    r"""
    #include <legate/runtime/detail/runtime.h>

    namespace {

    void *_get_cuda_stream()
    {
      return legate::detail::Runtime::get_runtime().get_cuda_stream();
    }

    } // namespace
    """
    void *_get_cuda_stream() except+


cdef (void *, bool) get_dlpack_tensor(object capsule):
    cdef void *ret = NULL

    if PyCapsule_IsValid(capsule, VERSIONED_CAPSULE_NAME):
        ret = PyCapsule_GetPointer(capsule, VERSIONED_CAPSULE_NAME)
        PyCapsule_SetName(capsule, VERSIONED_CAPSULE_NAME_USED)
        return ret, True

    if PyCapsule_IsValid(capsule, CAPSULE_NAME):
        ret = PyCapsule_GetPointer(capsule, CAPSULE_NAME)
        PyCapsule_SetName(capsule, CAPSULE_NAME_USED)
        return ret, False

    m = (
        "A DLPack tensor object cannot be consumed multiple times "
        f"(or object was not a DLPack capsule). Got: {capsule!r}"
    )
    raise ValueError(m)


cdef object get_stream(object device):
    if device is None:
        return None

    cdef DLDeviceType device_type = device[0]

    cdef tuple HOST_TYPES = (
        DLDeviceType.kDLCPU,
        DLDeviceType.kDLCUDAHost,
    )

    if device_type in HOST_TYPES:
        return None

    cdef void *cu_stream = NULL
    cdef tuple CUDA_TYPES = (
        DLDeviceType.kDLCUDA,
        DLDeviceType.kDLCUDAManaged,
    )

    if device_type in CUDA_TYPES:
        cu_stream = _get_cuda_stream()
        if cu_stream == NULL:
            return None  # legacy default stream
        return <uintptr_t>cu_stream

    m = f"Unhandled DLPack device type: {device_type}"
    raise BufferError(m)


def from_dlpack(
    object x,
    /,
    *,
    object device = None,
    object copy = None
) -> LogicalStore:
    if (device is None) and hasattr(x, "__dlpack_device__"):
        device = x.__dlpack_device__()

    cdef object stream = get_stream(device)
    # We don't check for attribute presence. The AttributeError that may be
    # raised below is part of the DLPack Python API standard:
    # https://data-apis.org/array-api/latest/API_specification/generated/array_api.from_dlpack.html#array_api.from_dlpack
    try:
        capsule = x.__dlpack__(
            stream=stream,
            max_version=(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
            dl_device=device,
            copy=copy
        )
    except TypeError:
        # TypeError may be thrown if the producer does not support one of the
        # keyword arguments:
        #
        # TypeError: __dlpack__() got an unexpected keyword argument
        # 'max_version'
        #
        # So we try again, only this time we pass in exactly what the user gave
        # us so that they can correct the error on their end.
        capsule = x.__dlpack__(dl_device=device, copy=copy)

    cdef void *tensor = NULL
    cdef bool  is_versioned = False

    tensor, is_versioned = get_dlpack_tensor(capsule)

    cdef _LogicalStore store

    with nogil:
        if is_versioned:
            ver_ptr = <_DLManagedTensorVersioned *>tensor
            store = _from_dlpack(&ver_ptr)
        else:
            unver_ptr = <_DLManagedTensor *>tensor
            store = _from_dlpack(&unver_ptr)

    return LogicalStore.from_handle(std_move(store))
