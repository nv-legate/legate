# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any, TypeVar

from ..._lib.utilities.typedefs import VariantCode

_T = TypeVar("_T")

def validate_variant(kind: VariantCode) -> None: ...
def dynamic_docstring(**kwargs: Any) -> Callable[[_T], _T]: ...
