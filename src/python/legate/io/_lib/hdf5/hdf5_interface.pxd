# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ....core._ext.cython_libcpp.string_view cimport std_string_view
from ....core._lib.data.logical_array cimport LogicalArray, _LogicalArray


cdef extern from "legate/io/hdf5/interface.h" namespace "legate" nogil:
    # These std_string_view arguments are in reality std::filesystem::path, so
    # these prototypes are actually a lie. But:
    #
    # 1. std_string_view automatically coerces into std::filesystem::path.
    # 2. We know how to automatically convert python strings into
    #    std_string_view.
    #
    # So this gives us the best of both worlds.
    cdef _LogicalArray _from_file "legate::io::hdf5::from_file" (
        std_string_view, std_string_view
    ) except+

    cdef void _to_file "legate::io::hdf5::to_file" (
        const _LogicalArray&, std_string_view, std_string_view
    ) except+


cpdef LogicalArray from_file(object path, str dataset_name)
cdef void _logical_array_to_file(
    LogicalArray array,
    object path,
    str dataset_name
)
