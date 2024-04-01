# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any

from ..mapping.machine import Machine
from .exception_mode import ExceptionMode

class Scope:
    def __init__(
        self,
        *,
        priority: int | None = None,
        exception_mode: ExceptionMode | None = None,
        provenance: str | None = None,
        machine: Machine | None = None,
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
