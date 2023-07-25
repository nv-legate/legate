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

from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec

class Magics:
    def __init__(self, shell: Any) -> None: ...

R = TypeVar("R")
P = ParamSpec("P")

line_magic: Callable[[Callable[P, R]], Callable[P, R]]
magics_class: Callable[[Callable[P, R]], Callable[P, R]]
