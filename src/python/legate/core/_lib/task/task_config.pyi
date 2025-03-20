# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..utilities.typedefs import LocalTaskID
from .variant_options import VariantOptions

class TaskConfig:
    def __init__(
        self, task_id: LocalTaskID, *, options: VariantOptions | None = None
    ) -> None: ...
    @property
    def task_id(self) -> LocalTaskID: ...
    @property
    def variant_options(self) -> VariantOptions | None: ...
    @variant_options.setter
    def variant_options(self, options: VariantOptions) -> None: ...
