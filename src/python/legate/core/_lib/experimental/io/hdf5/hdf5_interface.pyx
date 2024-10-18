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

from libcpp.utility cimport move as std_move

from ....._ext.cython_libcpp.string_view cimport (
    string_view as std_string_view,
    string_view_from_py as std_string_view_from_py,
)
from ....data.logical_array cimport LogicalArray, _LogicalArray


cpdef LogicalArray from_file(object path, str dataset_name):
    r"""Read an HDF5 array from disk using the native hdf5 API.

    This doesn't use KvikIO or GDS, instead a bounce buffer is used when
    running on a GPU.

    Parameters
    ----------
    path : Pathlike
        Path to the hdf5 file.
    dataset_name : str
        Name/path of the dataset. This must reference a single array, thus
        make sure to use the full path to the array inside the HDF5 file.

    Returns
    -------
    LogicalArray
        The Legate array read from disk.
    """
    cdef std_string_view cpp_path
    cdef std_string_view cpp_dataset_name

    cpp_path = std_string_view_from_py(str(path))
    cpp_dataset_name = std_string_view_from_py(dataset_name)

    cdef _LogicalArray ret

    with nogil:
        ret = _from_file(cpp_path, cpp_dataset_name)

    return LogicalArray.from_handle(std_move(ret))
