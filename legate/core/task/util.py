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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .type import VariantKind, VariantList

KNOWN_VARIANTS: set[VariantKind] = {"cpu", "gpu", "omp"}

DEFAULT_VARIANT_LIST: VariantList = ("cpu",)


def validate_variant(kind: str) -> None:
    r"""Confirm that a variant kind is one of the known variants.

    Parameters
    ----------
    kind : str
        The variant kind in string form.

    Raises
    ------
    ValueError
        If ``kind`` is an unknown variant kind.
    """
    if kind not in KNOWN_VARIANTS:
        raise ValueError(f"Unknown variant kind: {kind}")
