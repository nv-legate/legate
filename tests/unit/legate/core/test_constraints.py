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

import re
import sys
from collections.abc import Collection

import pytest

import legate.core as lc
from legate.core import (
    AutoTask,
    LogicalStore,
    Scalar,
    get_legate_runtime,
    types as ty,
)
from legate.core._lib.partitioning.constraint import (
    Constraint,
    ConstraintProxy,
    Variable,
)
from legate.core.task import InputStore, task


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


class TestAlign:
    def test_create_from_str(self) -> None:
        constraint = lc.align("x", "y")
        assert isinstance(constraint, ConstraintProxy)
        assert hasattr(constraint, "func")
        assert callable(constraint.func)
        assert constraint.func == lc.align
        assert hasattr(constraint, "args")
        assert constraint.args == ("x", "y")

    def test_create_from_variable(
        self, variable_x: Variable, variable_y: Variable
    ) -> None:
        constraint = lc.align(variable_x, variable_y)
        assert isinstance(constraint, Constraint)
        # Currently the only exposed python methods to check...
        expected_re = re.compile(
            r"Align\(X0{dummy_task\.<locals>\.foo:\d+}, "
            r"X1{dummy_task\.<locals>\.foo:\d+}\)"
        )
        assert expected_re.match(str(constraint)) is not None

    def test_create_bad(self, variable_x: Variable) -> None:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'rhs' has incorrect type (expected "
                "legate.core._lib.partitioning.constraint.Variable, got str)"
            ),
        ):
            lc.align(variable_x, "asdasd")  # type: ignore[call-overload]

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'rhs' has incorrect type (expected str, got "
                "legate.core._lib.partitioning.constraint.Variable)"
            ),
        ):
            lc.align("asdasd", variable_x)  # type: ignore[call-overload]


AXES: tuple[Collection[int], ...] = (
    tuple(),
    [],
    (1,),
    [1],
    (1, 2, 3),
    [3, 4, 5],
)


class TestBroadcast:
    @pytest.mark.parametrize("axes", AXES)
    def test_create_from_str(self, axes: Collection[int]) -> None:
        constraint = lc.broadcast("x", axes=axes)
        assert isinstance(constraint, ConstraintProxy)
        assert hasattr(constraint, "func")
        assert callable(constraint.func)
        assert constraint.func == lc.broadcast
        assert hasattr(constraint, "args")
        assert constraint.args == ("x", axes)

    @pytest.mark.parametrize("axes", AXES)
    def test_create_from_variable(
        self, variable_x: Variable, axes: Collection[int]
    ) -> None:
        constraint = lc.broadcast(variable_x, axes=axes)
        assert isinstance(constraint, Constraint)
        # Currently the only exposed python methods to check...
        if axes is None or not len(axes):
            expected_re = r"Broadcast\(X0{dummy_task\.<locals>\.foo:\d+}\)"
        else:
            ax_str = ",".join(map(str, axes))
            expected_re = (
                r"Broadcast\(X0{dummy_task\.<locals>\.foo:\d+}, "
                rf"\({ax_str}\)\)"
            )
        assert re.match(expected_re, str(constraint)) is not None

    def test_create_bad(self, variable_x: Variable) -> None:
        with pytest.raises(
            ValueError, match=re.escape("axes must be iterable")
        ):
            # Thanks for the warning mypy, but thats the point!
            lc.broadcast(variable_x, 1)  # type: ignore[call-overload]

        with pytest.raises(
            ValueError, match=re.escape("axes must be iterable")
        ):
            lc.broadcast("x", 1)  # type: ignore[call-overload]


class TestImage:
    def test_create_from_str(self) -> None:
        constraint = lc.image("x", "y")
        assert isinstance(constraint, ConstraintProxy)
        assert hasattr(constraint, "func")
        assert callable(constraint.func)
        assert constraint.func == lc.image
        assert hasattr(constraint, "args")
        assert constraint.args == ("x", "y")

    def test_create_from_variable(
        self, variable_x: Variable, variable_y: Variable
    ) -> None:
        constraint = lc.image(variable_x, variable_y)
        assert isinstance(constraint, Constraint)
        # Currently the only exposed python methods to check...
        expected_re = re.compile(
            r"ImageConstraint\(X0{dummy_task\.<locals>\.foo:\d+}, "
            r"X1{dummy_task\.<locals>\.foo:\d+}\)"
        )
        assert expected_re.match(str(constraint)) is not None

    def test_create_bad(self, variable_x: Variable) -> None:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_range' has incorrect type (expected "
                "legate.core._lib.partitioning.constraint.Variable, got str)"
            ),
        ):
            lc.image(variable_x, "asdasd")  # type: ignore[call-overload]

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_range' has incorrect type (expected str, got "
                "legate.core._lib.partitioning.constraint.Variable)"
            ),
        ):
            lc.image("asdasd", variable_x)  # type: ignore[call-overload]


FACTORS: tuple[tuple[int, ...], ...] = (tuple(), (1,), (2, 3, 4))


class TestScale:
    @pytest.mark.parametrize("factors", FACTORS)
    def test_create_from_str(self, factors: tuple[int, ...]) -> None:
        constraint = lc.scale(factors, "x", "y")
        assert isinstance(constraint, ConstraintProxy)
        assert hasattr(constraint, "func")
        assert callable(constraint.func)
        assert constraint.func == lc.scale
        assert hasattr(constraint, "args")
        assert constraint.args == (factors, "x", "y")

    @pytest.mark.parametrize("factors", FACTORS)
    def test_create_from_variable(
        self,
        factors: tuple[int, ...],
        variable_x: Variable,
        variable_y: Variable,
    ) -> None:
        constraint = lc.scale(factors, variable_x, variable_y)
        assert isinstance(constraint, Constraint)
        # Currently the only exposed python methods to check...
        factor_str = ",".join(map(str, factors))
        expected_re = re.compile(
            rf"ScaleConstraint\(\({factor_str}\), "
            r"X0{dummy_task\.<locals>\.foo:\d+}, "
            r"X1{dummy_task\.<locals>\.foo:\d+}\)"
        )
        assert expected_re.match(str(constraint)) is not None

    @pytest.mark.parametrize("factors", FACTORS)
    def test_create_bad(
        self, factors: tuple[int, ...], variable_x: Variable
    ) -> None:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_bigger' has incorrect type (expected "
                "legate.core._lib.partitioning.constraint.Variable, got str)"
            ),
        ):
            lc.scale(
                factors, variable_x, "asdasd"  # type: ignore[call-overload]
            )

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_bigger' has incorrect type (expected str, got "
                "legate.core._lib.partitioning.constraint.Variable)"
            ),
        ):
            lc.scale(
                factors, "asdasd", variable_x  # type: ignore[call-overload]
            )


OFFSETS: tuple[tuple[int, ...], ...] = (tuple(), (1,), (2, 3, 4))


class TestBloat:
    @pytest.mark.parametrize("lo_offsets", OFFSETS)
    @pytest.mark.parametrize("hi_offsets", OFFSETS)
    def test_create_from_str(
        self, lo_offsets: tuple[int, ...], hi_offsets: tuple[int, ...]
    ) -> None:
        constraint = lc.bloat("x", "y", lo_offsets, hi_offsets)
        assert isinstance(constraint, ConstraintProxy)
        assert hasattr(constraint, "func")
        assert callable(constraint.func)
        assert constraint.func == lc.bloat
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
        constraint = lc.bloat(variable_x, variable_y, lo_offsets, hi_offsets)
        assert isinstance(constraint, Constraint)
        # Currently the only exposed python methods to check...
        lo_str = ",".join(map(str, lo_offsets))
        hi_str = ",".join(map(str, hi_offsets))
        expected_re = re.compile(
            r"BloatConstraint\(X0{dummy_task\.<locals>\.foo:\d+}, "
            r"X1{dummy_task\.<locals>\.foo:\d+}, "
            rf"low: \({lo_str}\), "
            rf"high: \({hi_str}\)\)"
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
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_bloat' has incorrect type (expected "
                "legate.core._lib.partitioning.constraint.Variable, got str)"
            ),
        ):
            lc.bloat(  # type: ignore[call-overload]
                variable_x, "asdasd", lo_offsets, hi_offsets
            )

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument 'var_bloat' has incorrect type (expected str, got "
                "legate.core._lib.partitioning.constraint.Variable)"
            ),
        ):
            lc.bloat(  # type: ignore[call-overload]
                "asdasd", variable_x, lo_offsets, hi_offsets
            )


if __name__ == "__main__":
    sys.exit(pytest.main())
