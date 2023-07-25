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

import os
from typing import Any, Union

from ..install_info import header, libpath  # type: ignore [import]
from .legate import Library

# This is annoying but install_info is not present on unbuilt source, but is
# present in built source. So we either get an unfollowed-import error, or an
# unused-ignore error. Allow unused-ignores just in this file to work around
# mypy: warn-unused-ignores=False


class CoreLib(Library):
    def __init__(self) -> None:
        super().__init__()
        self._lib: Union[Any, None] = None

    def get_name(self) -> str:
        return "legate.core"

    def get_shared_library(self) -> str:
        libname = "liblgcore" + self.get_library_extension()
        return os.path.join(libpath, libname)

    def get_c_header(self) -> str:
        return header

    def initialize(self, shared_lib: Any) -> None:
        self._lib = shared_lib
        shared_lib.legate_parse_config()

    def get_registration_callback(self) -> str:
        return "legate_core_perform_registration"

    def destroy(self) -> None:
        if not self._lib:
            raise RuntimeError("CoreLib was never initialized")
        self._lib.legate_shutdown()


core_library = CoreLib()
