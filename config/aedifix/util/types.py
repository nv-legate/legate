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

import functools
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")
_U = TypeVar("_U")


def copy_method_signature(
    source: Callable[Concatenate[Any, _P], _T]
) -> Callable[[Callable[..., _U]], Callable[Concatenate[Any, _P], _U]]:
    r"""Copy a class methods signature.

    Paramaters
    ----------
    source : Callable
        The method to copy the signature from.

    Returns
    -------
    wrapped : Callable
        The destination callable, now matching the source's injected type
        signature.

    Notes
    -----
    `source` must be a class method, i.e. a callable whose first parameter is
    an implicit class object. `classmethod`s are also supported, but
    `staticmethod`s are not since those artificially remove the first parameter
    to the function.

    `target` must similarly also be a class method.

    To wrap standalone functions, see `copy_callable_signature`.

    This decorator has no impact on the actual runtime signature of the target
    callable, so if it different from `source`, then this will neither error or
    fail.
    """

    def wrapper(
        target: Callable[..., _U]
    ) -> Callable[Concatenate[Any, _P], _U]:
        @functools.wraps(source)
        def wrapped(self: Any, /, *args: _P.args, **kwargs: _P.kwargs) -> _U:
            return target(self, *args, **kwargs)

        return wrapped

    return wrapper


def copy_callable_signature(
    source: Callable[_P, _T]
) -> Callable[[Callable[..., _U]], Callable[_P, _U]]:
    r"""Copy a callable objects' signature.

    Paramaters
    ----------
    source : Callable
        The callable to copy the signature from.

    Returns
    -------
    wrapped : Callable
        The destination callable, now matching the source's injected type
        signature.

    Notes
    -----
    Both `source` and `target` must be standalone callables, i.e. not class
    methods. To wrap class methods, see `copy_method_signature`.

    The same note about runtime signature impact as detailed in
    `copy_method_signature` also applies here.
    """

    def wrapper(target: Callable[..., _U]) -> Callable[_P, _U]:
        @functools.wraps(source)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _U:
            return target(*args, **kwargs)

        return wrapped

    return wrapper
