# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, int32_t

cdef extern from "legate/utilities/detail/dlpack/dlpack.h" nogil:
    ctypedef struct _DLManagedTensorVersioned "::DLManagedTensorVersioned":
        void (*deleter)(_DLManagedTensorVersioned *)

    ctypedef struct _DLManagedTensor "::DLManagedTensor":
        pass

    ctypedef struct _DLPackVersion "::DLPackVersion":
        uint32_t major
        uint32_t minor

    ctypedef enum DLDeviceType:
        kDLCPU
        kDLCUDA
        kDLCUDAHost
        kDLCUDAManaged

    ctypedef struct _DLDevice "::DLDevice":
        DLDeviceType device_type
        int32_t device_id

    int DLPACK_MAJOR_VERSION
    int DLPACK_MINOR_VERSION

cdef extern from * nogil:
    r"""
    namespace {

    constexpr const char *const VERSIONED_CAPSULE_NAME = "dltensor_versioned";
    constexpr const char *const VERSIONED_CAPSULE_NAME_USED =
      "used_dltensor_versioned";
    constexpr const char *const CAPSULE_NAME = "dltensor";
    constexpr const char *const CAPSULE_NAME_USED = "used_dltensor";

    } // namespace
    """
    const char* VERSIONED_CAPSULE_NAME
    const char* VERSIONED_CAPSULE_NAME_USED
    const char* CAPSULE_NAME
    const char* CAPSULE_NAME_USED
