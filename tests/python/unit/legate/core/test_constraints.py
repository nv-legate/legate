# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

import pytest

import legate.core as lg
from legate.core import (
    AutoTask,
    LogicalStore,
    Scalar,
    get_legate_runtime,
    types as ty,
)
from legate.core._lib.partitioning.constraint import (
    Constraint,
    DeferredConstraint,
    Variable,
)
from legate.core.task import InputStore, task

if TYPE_CHECKING:
    from collections.abc import Collection


@pytest.fixture
def dummy_task() -> AutoTask:
    @task
    def foo(x: InputStore, y: InputStore) -> None:
        pass

    return get_legate_runtime().create_auto_task(foo.library, foo.task_id)


def _make_input_store() -> LogicalStore:
    runtime = get_legate_runtime()
    store = runtime.create_store(ty.int64, shape=(10,))
    scalar = Scalar(12345, ty.int64)
    runtime.issue_fill(store, scalar)
    return store


@pytest.fixture
def input_store_x() -> LogicalStore:
    return _make_input_store()


@pytest.fixture
def input_store_y() -> LogicalStore:
    return _make_input_store()


@pytest.fixture
def variable_x(dummy_task: AutoTask, input_store_x: LogicalStore) -> Variable:
    return dummy_task.add_input(input_store_x)


@pytest.fixture
def variable_y(dummy_task: AutoTask, input_store_y: LogicalStore) -> Variable:
    return dummy_task.add_input(input_store_y)


def repr_type_without_class(obj: type) -> str:
    # repr(SomeType) -> "<class 'legate.asdasd.SomeType'>"
    #
    # We just want "legate.asdasd.SomeType"
    return repr(obj).removeprefix("<class '").removesuffix("'>")


class TestVariable:
    def test_variable_properties(self, variable_x: Variable) -> None:
        # for code coverage purposes
        assert repr(variable_x) == str(variable_x)

    def test_direct_construction(self) -> None:
        msg = "Variable objects must not be constructed directly"
        with pytest.raises(ValueError, match=msg):
            Variable()


class TestAlign:
    def test_empty_args(self) -> None:
        # align() with 0 or 1 variables returns empty list
        assert lg.align() == []
        assert lg.align("x") == []

    def test_invalid_type(self) -> None:
        # align() with invalid first argument type should raise TypeError
        with pytest.raises(TypeError, match=re.escape(repr(int))):
            lg.align(123, 456)  # type: ignore[call-overload]

    def test_create_from_str(self) -> None:
        constraint = lg.align("x", "y")
        assert len(constraint) == 1
        assert isinstance(constraint[0], DeferredConstraint)
        assert hasattr(constraint[0], "args")
        assert constraint[0].args == ("x", "y")

    def test_create_many_from_str(self) -> None:
        variables = ("x", "y", "z", "w")
        constraint = lg.align(*variables)
        assert len(constraint) == len(variables) - 1
        for v, c in zip(variables[1:], constraint, strict=True):
            assert isinstance(c, DeferredConstraint)
            assert hasattr(c, "args")
            assert c.args == (variables[0], v)

    def test_create_from_variable(
        self, variable_x: Variable, variable_y: Variable
    ) -> None:
        constraint = lg.align(variable_x, variable_y)
        assert len(constraint) == 1
        assert isinstance(constraint[0], Constraint)
        # Currently the only exposed python methods to check...
        expected_re = re.compile(
            r"Align\(X0{dummy_task\.<locals>\.foo:\d+}, "
            r"X1{dummy_task\.<locals>\.foo:\d+}\)"
        )
        assert expected_re.match(str(constraint[0])) is not None
        assert repr(constraint[0]) == str(constraint[0])

    def test_create_many_from_variable(
        self, variable_x: Variable, variable_y: Variable
    ) -> None:
        variables = (variable_x, variable_y, variable_x, variable_y)
        constraint = lg.align(*variables)
        assert len(constraint) == len(variables) - 1
        for v, c in zip(variables[1:], constraint, strict=True):
            expected_re = re.escape(rf"Align({variable_x}, {v})")
            assert re.match(expected_re, str(c)) is not None
            assert repr(c) == str(c)

    def test_create_bad(self, variable_x: Variable) -> None:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "All variables for alignment must be variables, not strings, "
                f"have ({variable_x}, 'asdasd')"
            ),
        ):
            lg.align(variable_x, "asdasd")  # type: ignore[call-overload]

        with pytest.raises(
            TypeError,
            match=re.escape(
                "All variables for alignment must be strings, not variables, "
                f"have ('asdasd', {variable_x})"
            ),
        ):
            lg.align("asdasd", variable_x)  # type: ignore[call-overload]


AXES: tuple[Collection[int], ...] = ((), [], (1,), [1], (1, 2, 3), [3, 4, 5])


class TestBroadcast:
    @pytest.mark.parametrize("axes", AXES)
    def test_create_from_str(self, axes: Collection[int]) -> None:
        constraint = lg.broadcast("x", axes)
        assert isinstance(constraint, list)
        assert len(constraint) == 1
        assert isinstance(constraint[0], DeferredConstraint)
        assert hasattr(constraint[0], "args")
        assert constraint[0].args == ("x", axes)

    def test_create_many_from_str(self) -> None:
        variables = ("x", "y", "z", "w")
        constraint = lg.broadcast(*variables)
        assert isinstance(constraint, list)
        assert len(constraint) == len(variables)
        for v, c in zip(variables, constraint, strict=True):
            assert isinstance(c, DeferredConstraint)
            assert hasattr(c, "args")
            assert c.args == (v, ())

    def test_create_many_from_str_and_axes(self) -> None:
        variables = ("x", ("y", [1, 2, 3]), "z", ("w", [4, 5, 6]), "a")
        constraint = lg.broadcast(*variables)
        assert isinstance(constraint, list)
        assert len(constraint) == len(variables)
        for v, c in zip(variables, constraint, strict=True):
            assert isinstance(c, DeferredConstraint)
            assert hasattr(c, "args")
            if isinstance(v, tuple):
                assert c.args == v
            else:
                assert c.args == (v, ())

    @pytest.mark.parametrize("axes", AXES)
    def test_create_from_variable(
        self, variable_x: Variable, axes: Collection[int]
    ) -> None:
        constraint = lg.broadcast(variable_x, axes)
        assert isinstance(constraint, list)
        assert len(constraint) == 1
        assert isinstance(constraint[0], Constraint)
        # Currently the only exposed python methods to check...
        if axes is None or not len(axes):
            expected_re = r"Broadcast\(X0{dummy_task\.<locals>\.foo:\d+}\)"
        else:
            ax_str = ", ".join(map(str, axes))
            expected_re = (
                r"Broadcast\(X0{dummy_task\.<locals>\.foo:\d+}, "
                rf"\[{ax_str}\]\)"
            )
        assert re.match(expected_re, str(constraint[0])) is not None

    def test_create_many_from_variable(
        self, variable_x: Variable, variable_y: Variable
    ) -> None:
        variables = (variable_x, variable_y, variable_x, variable_y)
        constraint = lg.broadcast(*variables)
        assert isinstance(constraint, list)
        assert len(constraint) == len(variables)
        for v, c in zip(variables, constraint, strict=True):
            assert isinstance(c, Constraint)
            expected_re = re.escape(f"Broadcast({v})")
            assert re.match(expected_re, str(c)) is not None

    def test_create_bad(self, variable_x: Variable) -> None:
        with pytest.raises(
            ValueError, match=re.escape("axes must be iterable")
        ):
            # Thanks for the warning mypy, but that's the point!
            lg.broadcast(variable_x, 1)  # type: ignore[call-overload]

        with pytest.raises(
            ValueError, match=re.escape("axes must be iterable")
        ):
            lg.broadcast("x", 1)  # type: ignore[call-overload]

    def test_create_bad_variadic_type(self, variable_x: Variable) -> None:
        # When using variadic signature, passing an invalid type should raise
        with pytest.raises(TypeError, match=re.escape(repr(int))):
            lg.broadcast(variable_x, variable_x, 123)  # type: ignore[call-overload]

        with pytest.raises(TypeError, match=re.escape(repr(int))):
            lg.broadcast("x", "y", 123)  # type: ignore[call-overload]


class TestImage:
    def test_create_from_str(self) -> None:
        constraint = lg.image("x", "y")
        assert isinstance(constraint, DeferredConstraint)
        assert hasattr(constraint, "args")
        assert constraint.args == ("x", "y", lg.ImageComputationHint.NO_HINT)

    def test_create_from_str_with_hint(self) -> None:
        constraint = lg.image("x", "y", lg.ImageComputationHint.FIRST_LAST)
        assert isinstance(constraint, DeferredConstraint)
        assert hasattr(constraint, "args")
        assert constraint.args == (
            "x",
            "y",
            lg.ImageComputationHint.FIRST_LAST,
        )

    def test_create_from_variable(
        self, variable_x: Variable, variable_y: Variable
    ) -> None:
        constraint = lg.image(variable_x, variable_y)
        assert isinstance(constraint, Constraint)
        # Currently the only exposed python methods to check...
        expected_re = re.compile(
            r"ImageConstraint\(X0{dummy_task\.<locals>\.foo:\d+}, "
            r"X1{dummy_task\.<locals>\.foo:\d+}\)"
        )
        assert expected_re.match(str(constraint)) is not None

    def test_create_bad(self, variable_x: Variable) -> None:
        # repr(Variable) -> "<class 'legate.asdasd.Variable'>"
        #
        # We just want "legate.asdasd.Variable"
        type_str = repr_type_without_class(Variable)
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_range' has incorrect type (expected "
                f"{type_str}, got str)"
            ),
        ):
            lg.image(variable_x, "asdasd")  # type: ignore[call-overload]

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_range' has incorrect type (expected str, got "
                f"{type_str})"
            ),
        ):
            lg.image("asdasd", variable_x)  # type: ignore[call-overload]


FACTORS: tuple[tuple[int, ...], ...] = ((), (1,), (2, 3, 4))


class TestScale:
    @pytest.mark.parametrize("factors", FACTORS)
    def test_create_from_str(self, factors: tuple[int, ...]) -> None:
        constraint = lg.scale(factors, "x", "y")
        assert isinstance(constraint, DeferredConstraint)
        assert hasattr(constraint, "args")
        assert constraint.args == (factors, "x", "y")

    @pytest.mark.parametrize("factors", FACTORS)
    def test_create_from_variable(
        self,
        factors: tuple[int, ...],
        variable_x: Variable,
        variable_y: Variable,
    ) -> None:
        constraint = lg.scale(factors, variable_x, variable_y)
        assert isinstance(constraint, Constraint)
        # Currently the only exposed python methods to check...
        factor_str = ", ".join(map(str, factors))
        expected_re = re.compile(
            rf"ScaleConstraint\(\[{factor_str}\], "
            r"X0{dummy_task\.<locals>\.foo:\d+}, "
            r"X1{dummy_task\.<locals>\.foo:\d+}\)"
        )
        assert expected_re.match(str(constraint)) is not None

    @pytest.mark.parametrize("factors", FACTORS)
    def test_create_bad(
        self, factors: tuple[int, ...], variable_x: Variable
    ) -> None:
        # repr(Variable) -> "<class 'legate.asdasd.Variable'>"
        #
        # We just want "legate.asdasd.Variable"
        type_str = repr_type_without_class(Variable)
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_bigger' has incorrect type (expected "
                f"{type_str}, got str)"
            ),
        ):
            lg.scale(
                factors,
                variable_x,
                "asdasd",  # type: ignore[call-overload]
            )

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_bigger' has incorrect type (expected str, got "
                f"{type_str})"
            ),
        ):
            lg.scale(
                factors,
                "asdasd",
                variable_x,  # type: ignore[call-overload]
            )

    def test_negative_factor(
        self, variable_x: Variable, variable_y: Variable
    ) -> None:
        # Negative factors trigger OverflowError -> ValueError conversion
        # in tuple_from_iterable (utils.pyx)
        with pytest.raises(
            ValueError, match=re.escape("Extent must be a positive number")
        ):
            lg.scale((-1,), variable_x, variable_y)


OFFSETS: tuple[tuple[int, ...], ...] = ((), (1,), (2, 3, 4))


class TestBloat:
    @pytest.mark.parametrize("lo_offsets", OFFSETS)
    @pytest.mark.parametrize("hi_offsets", OFFSETS)
    def test_create_from_str(
        self, lo_offsets: tuple[int, ...], hi_offsets: tuple[int, ...]
    ) -> None:
        constraint = lg.bloat("x", "y", lo_offsets, hi_offsets)
        assert isinstance(constraint, DeferredConstraint)
        assert hasattr(constraint, "args")
        assert constraint.args == ("x", "y", lo_offsets, hi_offsets)

    @pytest.mark.parametrize("lo_offsets", OFFSETS)
    @pytest.mark.parametrize("hi_offsets", OFFSETS)
    def test_create_from_variable(
        self,
        variable_x: Variable,
        variable_y: Variable,
        lo_offsets: tuple[int, ...],
        hi_offsets: tuple[int, ...],
    ) -> None:
        constraint = lg.bloat(variable_x, variable_y, lo_offsets, hi_offsets)
        assert isinstance(constraint, Constraint)
        # Currently the only exposed python methods to check...
        lo_str = ", ".join(map(str, lo_offsets))
        hi_str = ", ".join(map(str, hi_offsets))
        expected_re = re.compile(
            r"BloatConstraint\(X0{dummy_task\.<locals>\.foo:\d+}, "
            r"X1{dummy_task\.<locals>\.foo:\d+}, "
            rf"low: \[{lo_str}\], "
            rf"high: \[{hi_str}\]\)"
        )
        assert expected_re.match(str(constraint)) is not None

    @pytest.mark.parametrize("lo_offsets", OFFSETS)
    @pytest.mark.parametrize("hi_offsets", OFFSETS)
    def test_create_bad(
        self,
        variable_x: Variable,
        lo_offsets: tuple[int, ...],
        hi_offsets: tuple[int, ...],
    ) -> None:
        # repr(Variable) -> "<class 'legate.asdasd.Variable'>"
        #
        # We just want "legate.asdasd.Variable"
        type_str = repr_type_without_class(Variable)
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_bloat' has incorrect type (expected "
                f"{type_str}, got str)"
            ),
        ):
            lg.bloat(  # type: ignore[call-overload]
                variable_x, "asdasd", lo_offsets, hi_offsets
            )

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_bloat' has incorrect type (expected str, got "
                f"{type_str})"
            ),
        ):
            lg.bloat(  # type: ignore[call-overload]
                "asdasd", variable_x, lo_offsets, hi_offsets
            )


if __name__ == "__main__":
    sys.exit(pytest.main())
