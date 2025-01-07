# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Callable
from typing import Any, Final, TypeVar

from ..._lib.utilities.typedefs import VariantCode
from .type import VariantList

_T = TypeVar("_T")

KNOWN_VARIANTS: Final[set[VariantCode]] = ...

DEFAULT_VARIANT_LIST: Final[VariantList] = ...

def validate_variant(kind: VariantCode) -> None: ...
def dynamic_docstring(**kwargs: Any) -> Callable[[_T], _T]: ...
