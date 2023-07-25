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

from typing import Any

class KernelSpec:
    display_name: str
    metadata: dict[str, Any]

    def __init__(
        self,
        argv: list[str],
        env: dict[str, str],
        display_name: str,
        language: str,
        metadata: dict[str, Any],
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class NoSuchKernel(Exception): ...

class KernelSpecManager:
    def __init__(self, **kwargs: Any) -> None: ...
    def get_kernel_spec(self, kernel_name: str) -> KernelSpec: ...
    def install_kernel_spec(
        self, source_dir: str, kernel_name: str, user: bool, prefix: str | None
    ) -> None: ...
