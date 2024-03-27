#!/usr/bin/env python3
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

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ...util.callables import classify_callable, get_calling_function


def foo() -> Any:
    return get_calling_function()


class Foo:
    # type hinting these properly would just confuse the type hinters
    # (not to mention, recurse infinitely). So these are just any's.
    def method(self) -> Any:
        return foo()

    @classmethod
    def class_method(cls) -> Any:
        return foo()

    @property
    def prop(self) -> Any:
        return foo()

    def __call__(self) -> Any:
        return foo()


class TestGetCallingFunction:
    def test_bare_func(self) -> None:
        def bar() -> Callable[[], Any]:
            return foo()

        assert foo() == self.test_bare_func
        assert bar() == bar

    def test_class(self) -> None:
        inst = Foo()
        assert inst.method() == inst.method
        assert inst.class_method() == inst.class_method
        # Error: "Callable[[Foo], Any]" has no attribute "fget" [attr-defined]
        #
        # ... yes it obviously does you absolute dunce
        assert inst.prop == Foo.prop.fget  # type: ignore[attr-defined]
        assert inst() == inst.__call__


class TestClassifyCallable:
    def test_func(self) -> None:
        qualname, path, lineno = classify_callable(foo)
        assert qualname == "config.aedifix.tests.util.test_callables.foo"
        assert path == Path(__file__)
        assert lineno == 24  # Unfortunately a brittle test...

        qualname, path, lineno = classify_callable(Foo().method)
        assert (
            qualname == "config.aedifix.tests.util.test_callables.Foo.method"
        )
        assert path == Path(__file__)
        assert lineno == 31  # Unfortunately a brittle test...

        qualname, path, lineno = classify_callable(Foo.class_method)
        assert (
            qualname
            == "config.aedifix.tests.util.test_callables.Foo.class_method"
        )
        assert path == Path(__file__)
        assert lineno == 34  # Unfortunately a brittle test...

        prop_function = Foo.prop.fget  # type: ignore[attr-defined]
        qualname, path, lineno = classify_callable(prop_function)
        assert qualname == "config.aedifix.tests.util.test_callables.Foo.prop"
        assert path == Path(__file__)
        assert lineno == 38  # Unfortunately a brittle test...

        qualname, path, lineno = classify_callable(Foo().__call__)
        assert (
            qualname == "config.aedifix.tests.util.test_callables.Foo.__call__"
        )
        assert path == Path(__file__)
        assert lineno == 42  # Unfortunately a brittle test...


if __name__ == "__main__":
    sys.exit(pytest.main())
