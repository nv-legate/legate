# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest

from legate.core import TaskContext, VariantCode, get_legate_runtime
from legate.core.task import InputArray, InputStore, OutputArray, task

from .util.task_util import make_input_array


def qual_name(obj: type) -> str:
    return f"{obj.__module__}.{obj.__name__}"


class TestTaskContext:
    def test_basic(self) -> None:
        @task(variants=(VariantCode.CPU,))
        def foo(ctx: TaskContext) -> None:
            assert isinstance(ctx, TaskContext)
            assert ctx.get_variant_kind() == VariantCode.CPU
            assert ctx.inputs == ()
            assert ctx.outputs == ()
            assert ctx.reductions == ()
            assert ctx.scalars == ()
            assert ctx.is_single_task() is True
            assert ctx.launch_domain == ((0,), (0,))
            assert ctx.task_stream is None
            assert ctx.can_raise_exception() is False

        foo()
        get_legate_runtime().issue_execution_fence(block=True)

    def test_multi_arg(self) -> None:
        @task
        def foo(ctx: TaskContext, x: InputArray, y: OutputArray) -> None:
            assert isinstance(ctx, TaskContext)
            assert ctx.get_variant_kind() == VariantCode.CPU
            assert ctx.inputs == (x,)
            assert ctx.outputs == (y,)
            assert ctx.reductions == ()
            assert ctx.scalars == ()
            assert ctx.is_single_task() is True
            assert ctx.launch_domain == ((0,), (0,))
            assert ctx.task_stream is None
            assert ctx.can_raise_exception() is False

        x = make_input_array()
        foo(x=x, y=x)

    def test_wrong_position(self) -> None:
        def foo(x: InputStore, ctx: TaskContext) -> None:
            pass

        msg = re.escape(
            "Explicit task context argument must appear as the first argument "
            "to the task. Found it in position 2: "
            f"(x: {qual_name(InputStore)}, ctx: {qual_name(TaskContext)}) "
            "-> None."
        )
        with pytest.raises(TypeError, match=msg):
            task(foo)

    def test_no_default_args(self) -> None:
        def foo(context: TaskContext | None = None) -> None:
            pass

        msg = re.escape(
            "Explicit task context argument must not have a default value "
            f"(found 'context: {qual_name(TaskContext)} | None = None'). "
            "Task context arguments are passed unconditionally to the task "
            "if requested, so it will never take the default value."
        )
        with pytest.raises(TypeError, match=msg):
            task(foo)
