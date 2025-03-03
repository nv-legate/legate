# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import CodeType, FrameType

_get_calling_function_called = 0


class GetCallingFuncRecursionError(Exception):
    pass


# regular function or staticmethod
def _gcf_function(fr: FrameType, co: CodeType) -> Callable[..., Any]:
    return fr.f_globals[co.co_name]


def _gcf_method(fr: FrameType, co: CodeType) -> Callable[..., Any]:
    obj = fr.f_locals["self"]
    name = co.co_name
    try:
        return getattr(obj, name)
    except GetCallingFuncRecursionError:
        # it's possible that the "function" is actually a property, e.g.:
        #
        # class Foo:
        #     @property
        #     def foo(self):
        #         return get_calling_function()
        #
        # In which case this would infinitely recurse. So we have to bypass the
        # descriptor, and try and retrieve the actual
        # function
        prop = inspect.getattr_static(obj, name)
        if (
            isinstance(prop, property)
            and (prop_getter := prop.fget) is not None
        ):
            return prop_getter
        # Another game is afoot...
        raise


def _gcf_classmethod(fr: FrameType, co: CodeType) -> Callable[..., Any]:
    return getattr(fr.f_locals["cls"], co.co_name)


def _gcf_get_local(fr: FrameType, name: str) -> Callable[..., Any]:
    f_back = fr.f_back
    # Item "None" of "FrameType | None" has no attribute "f_locals"
    # [union-attr]
    return f_back.f_locals[name]  # type: ignore[union-attr]


def _gcf_nested(fr: FrameType, co: CodeType) -> Callable[..., Any]:
    return _gcf_get_local(fr, co.co_name)


def _gcf_functools_wraps(fr: FrameType, _co: CodeType) -> Callable[..., Any]:
    return _gcf_get_local(fr, "func")


def _gcf_misc_1(fr: FrameType, _co: CodeType) -> Callable[..., Any]:
    return _gcf_get_local(fr, "meth")


def _gcf_misc_2(fr: FrameType, _co: CodeType) -> Callable[..., Any]:
    return _gcf_get_local(fr, "f")


def _get_calling_function_impl() -> Callable[..., Any]:
    stack = inspect.stack()
    maxidx = 20
    for idx in range(
        3,  # the caller of the function that called get_calling_function()
        maxidx,  # if anyone is more than 20 ignores deep, probably a bug
    ):
        try:
            fr = stack[idx].frame
        except IndexError:
            break
        co = fr.f_code

        getters = (
            _gcf_function,
            _gcf_method,
            _gcf_classmethod,
            _gcf_nested,
            _gcf_functools_wraps,
            _gcf_misc_1,
            _gcf_misc_2,
        )
        for getter in getters:
            try:
                func = getter(fr, co)
            except (KeyError, AttributeError):
                continue
            if func.__code__ != co:
                continue
            if getattr(func, "__config_log_ignore___", False):
                # found a passthrough function, continue searching up the stack
                break
            return func
        else:
            # we did not break due to ignores, so we failed to find the caller
            break
    else:
        # We exhausted the range iterator
        msg = (
            f"Iterated {maxidx} times trying to determine the calling "
            "function, but failed to find it. This is likely a bug! "
            f"Stack: {stack}"
        )
        raise AssertionError(msg)
    raise ValueError


def get_calling_function() -> Callable[..., Any]:
    r"""Finds the calling function in many decent cases.

    Returns
    -------
    func : Any
        The function or method object that called this function.

    Raises
    ------
    ValueError
        If the calling function cannot be determined.
    """
    global _get_calling_function_called  # noqa: PLW0603

    if _get_calling_function_called:
        raise GetCallingFuncRecursionError

    _get_calling_function_called += 1
    try:
        return _get_calling_function_impl()
    finally:
        _get_calling_function_called -= 1
        assert _get_calling_function_called >= 0


def _is_classmethod(method: Any) -> bool:
    bound_to = getattr(method, "__self__", None)
    if not isinstance(bound_to, type):
        # must be bound to a class
        return False
    name = method.__name__
    for cls in bound_to.__mro__:
        descriptor = vars(cls).get(name)
        if descriptor is not None:
            return isinstance(descriptor, classmethod)
    return False


@lru_cache
def classify_callable(
    fn: Callable[..., Any], *, fully_qualify: bool = True
) -> tuple[str, Path, int]:
    r"""Classify a callable object.

    Parameters
    ----------
    fn : Callable
        The callable object to classify.
    fully_qualify : bool, True
        Whether to return the fully qualified name, or just the short name
        of `fn`.

    Returns
    -------
    qualname : str
        The qualified name of `fn`.
    src_file : Path
        The full path to the source file where `fn` was defined.
    lineno : int
        The line number in `src_file` where `fn` was defined.

    Raises
    ------
    TypeError
        If `fn` is not a callable object, or not handled by this method.
    """
    if inspect.ismethod(fn):  # method or classmethod
        if _is_classmethod(fn):
            class_obj = fn.__self__
        else:
            class_obj = fn.__self__.__class__
        assert hasattr(class_obj, "__name__")  # appease mypy
        if fully_qualify:
            qualname = (
                f"{class_obj.__module__}.{class_obj.__name__}.{fn.__name__}"
            )
        else:
            qualname = f"{class_obj.__name__}.{fn.__name__}"
        func_obj = fn.__func__
    elif inspect.isfunction(fn):
        module = inspect.getmodule(fn)
        assert module is not None, (
            f"Could not determine host module for function {fn}"
        )
        if fully_qualify:
            qualname = f"{module.__name__}.{fn.__qualname__}"
        else:
            qualname = fn.__qualname__
        func_obj = fn
    else:
        raise TypeError(fn)

    src_file = inspect.getsourcefile(fn)
    assert src_file is not None, (
        f"Could not determine source file for function {fn}"
    )
    lineno = func_obj.__code__.co_firstlineno
    return qualname, Path(src_file), lineno
