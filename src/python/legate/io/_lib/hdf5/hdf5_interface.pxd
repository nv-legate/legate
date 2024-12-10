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

from ....core._ext.cython_libcpp.string_view cimport (
    string_view as std_string_view,
)
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


cpdef LogicalArray from_file(object path, str dataset_name)
