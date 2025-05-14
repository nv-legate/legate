# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ..mapping.machine import Machine
from ..runtime.exception_mode import ExceptionMode
from .parallel_policy import ParallelPolicy

class Scope:
    def __init__(
        self,
        *,
        priority: int | None = None,
        exception_mode: ExceptionMode | None = None,
        provenance: str | None = None,
        machine: Machine | None = None,
        parallel_policy: ParallelPolicy | None = None,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, _: Any, __: Any, ___: Any) -> None: ...
    @staticmethod
    def priority() -> int: ...
    @staticmethod
    def exception_mode() -> ExceptionMode: ...
    @staticmethod
    def provenance() -> str: ...
    @staticmethod
    def machine() -> Machine: ...
    @staticmethod
    def parallel_policy() -> ParallelPolicy: ...
