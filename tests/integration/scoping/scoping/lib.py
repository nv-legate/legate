# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from typing import Any

from legate.core import Library, get_legate_runtime


class UserLibrary(Library):
    def __init__(self, name: str) -> None:
        self.name = name
        self.shared_object: Any = None

    @property
    def cffi(self) -> Any:
        return self.shared_object

    def get_name(self) -> str:
        return self.name

    def get_shared_library(self) -> str:
        from scoping.install_info import libpath

        return os.path.join(
            libpath, f"libscoping{self.get_library_extension()}"
        )

    def get_c_header(self) -> str:
        from scoping.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "perform_registration"

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass


user_lib = UserLibrary("scoping")
user_context = get_legate_runtime().register_library(user_lib)
