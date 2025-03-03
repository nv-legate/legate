# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

    runtime = get_legate_runtime()
    return runtime.create_auto_task(runtime.core_library, foo.task_id)


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


class TestAlign:
    def test_create_from_str(self) -> None:
        constraint = lg.align("x", "y")
        assert isinstance(constraint, DeferredConstraint)
        assert hasattr(constraint, "args")
        assert constraint.args == ("x", "y")

    def test_create_from_variable(
        self, variable_x: Variable, variable_y: Variable
    ) -> None:
        constraint = lg.align(variable_x, variable_y)
        assert isinstance(constraint, Constraint)
        # Currently the only exposed python methods to check...
        expected_re = re.compile(
            r"Align\(X0{dummy_task\.<locals>\.foo:\d+}, "
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
                "Argument 'rhs' has incorrect type (expected "
                f"{type_str}, got str)"
            ),
        ):
            lg.align(variable_x, "asdasd")  # type: ignore[call-overload]

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'rhs' has incorrect type (expected str, got "
                f"{type_str})"
            ),
        ):
            lg.align("asdasd", variable_x)  # type: ignore[call-overload]


AXES: tuple[Collection[int], ...] = ((), [], (1,), [1], (1, 2, 3), [3, 4, 5])


class TestBroadcast:
    @pytest.mark.parametrize("axes", AXES)
    def test_create_from_str(self, axes: Collection[int]) -> None:
        constraint = lg.broadcast("x", axes=axes)
        assert isinstance(constraint, DeferredConstraint)
        assert hasattr(constraint, "args")
        assert constraint.args == ("x", axes)

    @pytest.mark.parametrize("axes", AXES)
    def test_create_from_variable(
        self, variable_x: Variable, axes: Collection[int]
    ) -> None:
        constraint = lg.broadcast(variable_x, axes=axes)
        assert isinstance(constraint, Constraint)
        # Currently the only exposed python methods to check...
        if axes is None or not len(axes):
            expected_re = r"Broadcast\(X0{dummy_task\.<locals>\.foo:\d+}\)"
        else:
            ax_str = ", ".join(map(str, axes))
            expected_re = (
                r"Broadcast\(X0{dummy_task\.<locals>\.foo:\d+}, "
                rf"\[{ax_str}\]\)"
            )
        assert re.match(expected_re, str(constraint)) is not None

    def test_create_bad(self, variable_x: Variable) -> None:
        with pytest.raises(
            ValueError, match=re.escape("axes must be iterable")
        ):
            # Thanks for the warning mypy, but thats the point!
            lg.broadcast(variable_x, 1)  # type: ignore[call-overload]

        with pytest.raises(
            ValueError, match=re.escape("axes must be iterable")
        ):
            lg.broadcast("x", 1)  # type: ignore[call-overload]


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
