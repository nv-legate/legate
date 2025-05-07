# Copyright 2024-2025 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import functools
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Self, cast

from ..._lib.experimental.io.kvikio.kvikio_interface import from_file, to_file

OpenFlag = Literal["r", "rb", "r+", "rb+", "w", "wb", "w+", "wb+"]

if TYPE_CHECKING:
    from ..._lib.data.logical_array import LogicalArray
    from ..._lib.type.types import Type


class FileHandle:
    """File handle for distributed IO."""

    supported_open_flags: Final = {"r", "r+", "w", "w+", "b"}

    def __init__(self, file: Path | str, flags: OpenFlag = "r"):
        """Open file for distributed IO.

        The file is opened in the constructor immediately and not in a
        Legate task. This means that re-opening a file that was created
        by a not-yet-executed Legate task requires a blocking fence like
        ``get_legate_runtime().issue_execution_fence(block=True)``.

        Additional limitations:
          - The file path to the file must not change while opened.
          - The file size can only be changed through this handle. That is,
            access to the file outside of this file handle is allowed as long
            as the file size doesn't change.
          - Writing to the file using a file handle (e.g. ``FileHandle.write``)
            is a deferred task thus in order to read the file at a later point,
            a blocking execution fence is required even when using the same
            file handle.

        Parameters
        ----------
        file: Path | str
            Path-like object giving the pathname (absolute or relative to the
            current working directory) of the file to be opened.
        flags: str, optional
            "r"  -> open for reading (default)
            "r+" -> open for reading and writing
            "w"  -> open for writing, truncating the file first
            "w+" -> open for reading and writing, truncating the file first
            "b"  -> ignored, the file is always opened in binary mode
        """
        flags = cast(OpenFlag, flags.replace("b", ""))
        if flags not in self.supported_open_flags:
            msg = f"Unsupported flags: '{flags}'"
            raise NotImplementedError(msg)
        self._closed = False
        self._filepath = Path(file)
        self._flags = flags

        # We open the file here in order to create or truncate files
        # opened in "w" mode, which is required because ``TaskOpCode.WRITE``
        # always opens the file in "r+" mode.
        with self._filepath.open(mode=flags):
            pass

    @functools.cached_property
    def size(self) -> int:
        """File size in bytes."""
        return self._filepath.stat().st_size

    def close(self) -> None:
        """Deregister the file and close the file."""
        self._closed = True

    @property
    def closed(self) -> bool:  # noqa: D102
        return self._closed

    def __enter__(self) -> Self:  # noqa: D105
        return self

    def __exit__(  # noqa: D105
        self, _exc_type: Any, _exc_val: Any, _exc_tb: Any
    ) -> None:
        self.close()

    def read(self, ty: Type) -> LogicalArray:
        r"""Reads specified buffer from the file into device or host memory.

        Parameters
        ----------
        ty : Type
            The type of the array to read.

        Returns
        -------
        LogicalArray
            The array read from disk.
        """
        if self._closed:
            msg = "file is closed"
            raise RuntimeError(msg)
        if "r" not in self._flags and "+" not in self._flags:
            msg = f"Cannot read a file opened with flags={self._flags}"
            raise ValueError(msg)

        return from_file(str(self._filepath), ty)

    def write(self, array: LogicalArray) -> None:
        r"""Writes specified buffer from device or host memory to the file.

        Hint, if a subsequent operation reads this file, insert a fence in
        between such as
        ``legate.core.get_legate_runtime().issue_execution_fence(block=False)``.

        Parameters
        ----------
        array : LogicalArray
            The array to write.
        """
        if self._closed:
            msg = "file is closed"
            raise RuntimeError(msg)
        if "w" not in self._flags and "+" not in self._flags:
            msg = f"Cannot write to a file opened with flags={self._flags}"
            raise ValueError(msg)
        with contextlib.suppress(AttributeError):
            del self.size  # clear the cached file size (if it exist)

        to_file(str(self._filepath), array)
