# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.utility cimport move as std_move

from cpython cimport PyCapsule_IsValid, PyCapsule_GetPointer, PyCapsule_SetName

from ....runtime.runtime cimport get_legate_runtime
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

cdef struct DLPackTensor:
    void* tensor
    bool is_versioned


cdef DLPackTensor get_dlpack_tensor(object capsule):
    cdef DLPackTensor ret

    if PyCapsule_IsValid(capsule, VERSIONED_CAPSULE_NAME):
        ret.tensor = PyCapsule_GetPointer(capsule, VERSIONED_CAPSULE_NAME)
        ret.is_versioned = True
        PyCapsule_SetName(capsule, VERSIONED_CAPSULE_NAME_USED)
        return ret

    if PyCapsule_IsValid(capsule, CAPSULE_NAME):
        ret.tensor = PyCapsule_GetPointer(capsule, CAPSULE_NAME)
        ret.is_versioned = False
        PyCapsule_SetName(capsule, CAPSULE_NAME_USED)
        return ret

    m = (
        "A DLPack tensor object cannot be consumed multiple times "
        f"(or object was not a DLPack capsule). Got: {capsule!r}"
    )
    raise ValueError(m)


cdef bool check_device_type(object device):
    """
    Check if the device is one of supported device types and returns ``True``
    if the device is GPU.
    """
    if device is None:
        return False

    cdef DLDeviceType device_type = device[0]
    cdef tuple HOST_TYPES = (
        DLDeviceType.kDLCPU,
        DLDeviceType.kDLCUDAHost,
    )
    cdef tuple GPU_DEVICE_TYPES = (
        DLDeviceType.kDLCUDA,
        DLDeviceType.kDLCUDAManaged,
    )

    if device_type in HOST_TYPES:
        return False

    if device_type in GPU_DEVICE_TYPES:
        return True

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

    is_device_gpu = check_device_type(device)
    # We don't check for attribute presence. The AttributeError that may be
    # raised below is part of the DLPack Python API standard:
    # https://data-apis.org/array-api/latest/API_specification/generated/array_api.from_dlpack.html#array_api.from_dlpack
    try:
        capsule = x.__dlpack__(
            # Passing a null stream as a consuming stream, as we don't know
            # which task will consume this store later and where it will run
            # (which is even less obvious if the store gets partitioned across
            # multiple GPUs).
            stream=None,
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

    if is_device_gpu:
        # Since we pass a null stream above to extract the capsule, we should
        # synchronize it here. Otherwise, non-blocking streams used in
        # downstream GPU tasks are not properly synchronized.
        get_legate_runtime().synchronize_cuda_stream(NULL)

    cdef DLPackTensor dl_tensor

    dl_tensor = get_dlpack_tensor(capsule)

    cdef _LogicalStore store

    with nogil:
        if dl_tensor.is_versioned:
            ver_ptr = <_DLManagedTensorVersioned *>dl_tensor.tensor
            store = _from_dlpack(&ver_ptr)
        else:
            unver_ptr = <_DLManagedTensor *>dl_tensor.tensor
            store = _from_dlpack(&unver_ptr)

    return LogicalStore.from_handle(std_move(store))
