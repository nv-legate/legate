# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Callable, ParamSpec, TypeVar

class Magics:
    def __init__(self, shell: Any) -> None: ...

_R = TypeVar("_R")
_P = ParamSpec("_P")

line_magic: Callable[[Callable[_P, _R]], Callable[_P, _R]]
magics_class: Callable[[Callable[_P, _R]], Callable[_P, _R]]
