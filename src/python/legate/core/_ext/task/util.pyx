# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Any, TypeVar

from ..._lib.utilities.typedefs cimport VariantCode

# needed for "kind not in VariantCode"

from ..._lib.utilities.typedefs import VariantCode as PyVariantCode


cdef object _T = TypeVar("_T")

cpdef void validate_variant(VariantCode kind):
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
    if kind not in PyVariantCode:
        raise ValueError(f"Unknown variant kind: {kind}")


def dynamic_docstring(**kwargs: Any) -> Callable[[_T], _T]:
    def wrapper(obj: _T) -> _T:
        if (obj_doc := getattr(obj, "__doc__", None)) is not None:
            assert isinstance(obj_doc, str)
            try:
                setattr(obj, "__doc__", obj_doc.format(**kwargs))
            except AttributeError:
                print(obj)
                assert 0
        return obj

    return wrapper
