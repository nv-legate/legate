# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .core.magic import Magics

__all__ = ("InteractiveShell",)

class InteractiveShell:
    def register_magics(self, *objs: Magics) -> None: ...
