# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
        m = f"Unknown variant kind: {kind}"  # pragma: no cover
        raise ValueError(m)  # pragma: no cover


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


cdef str _get_callable_name(object obj):
    try:
        return obj.__qualname__
    except AttributeError:
        pass
    try:
        return obj.__class__.__qualname__
    except AttributeError:
        return obj.__name__
