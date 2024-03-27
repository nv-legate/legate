# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import Any, Callable

# cheating a bit here, these are part of cffi but it is a mess
class CData:
    def __getitem__(self, idx: int) -> Any: ...
    def __setitem__(self, idx: int, value: Any) -> None: ...

class CType:
    cname: str

class FFI:
    NULL: CData

    def new(self, ctype: str, *args: Any) -> CData: ...
    def cast(self, ctype: str, value: Any) -> CData: ...
    def cdef(self, cstring: str) -> None: ...
    def dlopen(self, path: str) -> Any: ...
    def typeof(self, tpy: str | CData) -> CType: ...
    def addressof(self, value: CData) -> Any: ...
    def sizeof(self, value: Any) -> int: ...
    def from_buffer(self, value: CData | memoryview) -> Any: ...
    def buffer(self, value: CData, size: int = 0) -> Any: ...
    def unpack(self, value: CData, maxlen: int = 0) -> bytes: ...
    def gc(
        self, value: CData, destructor: Callable[[CData], None], size: int = 0
    ) -> CData: ...

ffi: FFI

is_legion_python: bool
