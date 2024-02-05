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
from __future__ import annotations

from typing import Callable, overload

from .py_task import PyTask
from .type import UserFunction, VariantList
from .util import DEFAULT_VARIANT_LIST, dynamic_docstring

@overload
def task(func: UserFunction) -> PyTask: ...
@overload
def task(
    *, variants: VariantList = DEFAULT_VARIANT_LIST, register: bool = True
) -> Callable[[UserFunction], PyTask]: ...
