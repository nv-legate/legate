# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

class ResourceConfig:
    def __init__(
        self,
        *,
        max_tasks: int | None = None,
        max_dyn_tasks: int | None = None,
        max_reduction_ops: int | None = None,
    ) -> None: ...
    @property
    def max_tasks(self) -> int: ...
    @max_tasks.setter
    def max_tasks(self, max_tasks: int) -> None: ...
    @property
    def max_dyn_tasks(self) -> int: ...
    @max_dyn_tasks.setter
    def max_dyn_tasks(self, max_dyn_tasks: int) -> None: ...
    @property
    def max_reduction_ops(self) -> int: ...
    @max_reduction_ops.setter
    def max_reduction_ops(self, max_reduction_ops: int) -> None: ...
