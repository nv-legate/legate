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

from collections.abc import Callable
from typing import Any, Final, TypeAlias, TypeVar

from .type import VariantKind, VariantList

_T = TypeVar("_T")

KNOWN_VARIANTS: Final[set[VariantKind]] = {"cpu", "gpu", "omp"}

DEFAULT_VARIANT_LIST: Final[VariantList] = ("cpu",)

def validate_variant(kind: VariantKind) -> None: ...
def dynamic_docstring(**kwargs: Any) -> Callable[[_T], _T]: ...
