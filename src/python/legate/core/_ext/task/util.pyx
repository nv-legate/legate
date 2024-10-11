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
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ..._lib.utilities.typedefs cimport VariantCode
from .type cimport VariantList


cdef set[VariantCode] _KNOWN_VARIANTS = {
    VariantCode.CPU, VariantCode.GPU, VariantCode.OMP
}
# for python export
KNOWN_VARIANTS = _KNOWN_VARIANTS

cdef VariantList DEFAULT_VARIANT_LIST = (VariantCode.CPU,)

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
    if kind not in _KNOWN_VARIANTS:
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
