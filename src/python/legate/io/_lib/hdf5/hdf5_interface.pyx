# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.utility cimport move as std_move

from ....core.data_interface import as_logical_array
from ....core._ext.cython_libcpp.string_view cimport (
    std_string_view,
    std_string_view_from_py,
)
from ....core._lib.data.logical_array cimport LogicalArray, _LogicalArray


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
    cdef str str_path = str(path)
    cdef std_string_view cpp_path
    cdef std_string_view cpp_dataset_name

    cpp_path = std_string_view_from_py(str_path)
    cpp_dataset_name = std_string_view_from_py(dataset_name)

    cdef _LogicalArray ret

    with nogil:
        ret = _from_file(cpp_path, cpp_dataset_name)

    return LogicalArray.from_handle(std_move(ret))


cpdef to_file(object array, object path, str dataset_name):
    r"""Write a LogicalArray to disk using HDF5.

    If ``path`` already exists at the time of writing, the file will be
    overwritten.

    ``path`` may be absolute or relative. If it is relative, it will be written
    relative to the current working directory at the time of this function call.

    ``path`` may not fully exist at the time of this function call. Any missing
    directories are created (with the same permissions and properties of the
    current process) before tasks are launched. However, no protection is
    provided if those directories are later deleted before the task executes -
    the tasks assume these directories exist when they execute.

    ``array`` must not be unbound.

    Parameters
    ----------
    array : LogicalArrayLike
        The array-like object to serialize.
    path : Pathlike
        Path to write to.
    dataset_name : str
        The name of the data set to store the array under.

    Raises
    ------
    ValueError
        If ``path`` would not be a valid path name, for example if it is a
        directory name. Generally speaking, it should be in the form
        ``/path/to/file.h5``.
    """
    ary = as_logical_array(array)
    _logical_array_to_file(ary, path, dataset_name)


cdef void _logical_array_to_file(
    LogicalArray array,
    object path, str
    dataset_name
):
    cdef str str_path = str(path)
    cdef std_string_view cpp_path
    cdef std_string_view cpp_dataset_name

    cpp_path = std_string_view_from_py(str_path)
    cpp_dataset_name = std_string_view_from_py(dataset_name)

    with nogil:
        _to_file(array._handle, cpp_path, cpp_dataset_name)
