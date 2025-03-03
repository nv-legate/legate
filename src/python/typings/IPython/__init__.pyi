# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .core.magic import Magics

class InteractiveShell:
    def register_magics(self, *objs: Magics) -> None: ...
