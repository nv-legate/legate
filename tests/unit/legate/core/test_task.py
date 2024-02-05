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

import random
import re
from typing import ParamSpec

import numpy as np
import numpy.testing
import pytest

import legate.core as lg
from legate.core import PhysicalStore, get_legate_runtime, task as lct
from legate.core.task import InputStore, OutputStore, PyTask, VariantInvoker

from .util.task_util import (
    UNTYPED_FUNCS,
    USER_FUNC_ARGS,
    USER_FUNCS,
    ArgDescr,
    FakeArray,
    FakeAutoTask,
    FakeScalar,
    FakeTaskContext,
    TestFunction,
    assert_isinstance,
    make_input_store,
    make_output_store,
    multi_input,
    multi_output,
    single_input,
)

_P = ParamSpec("_P")


@pytest.fixture
def fake_auto_task() -> FakeAutoTask:
    return FakeAutoTask()


class BaseTest:
    def check_valid_task(self, task: PyTask) -> None:
        assert isinstance(task, PyTask)
        assert callable(task)
        assert isinstance(task.registered, bool)

    def check_valid_registered_task(self, task: PyTask) -> None:
        self.check_valid_task(task)
        assert task.registered
        task_id = task.task_id
        assert isinstance(task_id, int)
        assert task_id > 0

    def check_valid_unregistered_task(self, task: PyTask) -> None:
        self.check_valid_task(task)
        assert not task.registered
        with pytest.raises(RuntimeError):
            task.task_id  # must complete registration first

    def check_valid_invoker(
        self, invoker: VariantInvoker, func: TestFunction[_P, None]
    ) -> None:
        assert callable(invoker)
        assert invoker.valid_signature(func)
        assert invoker.inputs == getattr(func, "inputs", ())
        assert invoker.outputs == getattr(func, "outputs", ())
        assert invoker.reductions == getattr(func, "reductions", ())
        assert invoker.scalars == getattr(func, "scalars", ())

    def check_func_called(self, func: TestFunction[_P, None]) -> None:
        get_legate_runtime().issue_execution_fence(block=True)
        assert func.called


class TestTask(BaseTest):
    @pytest.mark.parametrize("func", USER_FUNCS)
    @pytest.mark.parametrize("register", [True, False])
    def test_create_auto(
        self, func: TestFunction[_P, None], register: bool
    ) -> None:
        task = lct.task(register=register)(func)

        if not register:
            self.check_valid_unregistered_task(task)
            task.complete_registration()
        self.check_valid_registered_task(task)

    # This test is parameterized on checking whether the function was called
    # because we want to test race-condition scenarios, in particular as they
    # pertain to the GIL. In the case where we don't assert func.called we
    # also don't issue a fence, and hence multiple tasks may execute together
    # and potentially cause memory errors (if they misbehave).
    @pytest.mark.parametrize(
        "in_func, func_args", zip(USER_FUNCS, USER_FUNC_ARGS)
    )
    @pytest.mark.parametrize("register", [True, False])
    @pytest.mark.parametrize("check_called", [True, False])
    def test_executable_auto(
        self,
        in_func: TestFunction[_P, None],
        func_args: ArgDescr,
        register: bool,
        check_called: bool,
    ) -> None:
        # make a deep copy of func since tasks may execute out of order on
        # multiple threads which may clobber the called attribute
        func = in_func.deep_clone()

        task = lct.task(register=register)(func)

        if not register:
            self.check_valid_unregistered_task(task)
            with pytest.raises(RuntimeError):
                task(*func_args.args())

            task.complete_registration()

        self.check_valid_registered_task(task)
        assert not func.called
        task(*func_args.args())
        if check_called:
            self.check_func_called(func)

    @pytest.mark.parametrize(
        "in_func, func_args", zip(USER_FUNCS, USER_FUNC_ARGS)
    )
    @pytest.mark.parametrize("register", [True, False])
    @pytest.mark.parametrize("check_called", [True, False])
    def test_executable_prepare_call(
        self,
        in_func: TestFunction[_P, None],
        func_args: ArgDescr,
        register: bool,
        check_called: bool,
    ) -> None:
        # make a deep copy of func since tasks may execute out of order on
        # multiple threads which may clobber the called attribute
        func = in_func.deep_clone()

        task = lct.task(register=register)(func)

        if not register:
            self.check_valid_unregistered_task(task)
            with pytest.raises(RuntimeError):
                task(*func_args.args())

            task.complete_registration()

        self.check_valid_registered_task(task)
        assert not func.called
        task_inst = task.prepare_call(*func_args.args())

        dummy_store = make_input_store()
        with pytest.raises(
            RuntimeError,
            match=(
                "Attempting to add inputs to a prepared Python task is "
                "illegal!"
            ),
        ):
            task_inst.add_input(dummy_store, None)
        with pytest.raises(
            RuntimeError,
            match=(
                "Attempting to add outputs to a prepared Python task is "
                "illegal!"
            ),
        ):
            task_inst.add_output(dummy_store, None)
        with pytest.raises(
            RuntimeError,
            match=(
                "Attempting to add reductions to a prepared Python task is "
                "illegal!"
            ),
        ):
            task_inst.add_reduction(dummy_store, 0, None)
        with pytest.raises(
            RuntimeError,
            match=(
                "Attempting to add scalar arguments to a prepared Python task "
                "is illegal!"
            ),
        ):
            task_inst.add_scalar_arg(0, None)

        task_inst.execute()
        if check_called:
            self.check_func_called(func)

    def test_executable_wrong_arg_order(self) -> None:
        array_val = random.randint(0, 1000)
        c_val = random.randint(-1000, 1000)
        d_val = float(random.random() * c_val)

        def test_wrong_arg_order(
            a: InputStore, b: OutputStore, c: int, d: float
        ) -> None:
            assert_isinstance(a, PhysicalStore)
            assert (
                np.asarray(a.get_inline_allocation()).all()
                == np.array([array_val] * 10).all()
            )
            assert_isinstance(b, PhysicalStore)
            assert_isinstance(c, int)
            assert c == c_val
            assert_isinstance(d, float)
            assert d == d_val

        a = make_input_store(value=array_val)
        b = make_output_store()
        c = c_val
        d = d_val

        task = lct.task()(test_wrong_arg_order)
        # arguments correct, but kwargs in wrong
        task(d=d, c=c, b=b, a=a)

    @pytest.mark.parametrize(
        "in_func, func_args", zip(USER_FUNCS, USER_FUNC_ARGS)
    )
    def test_invoke_unhandled_args(
        self,
        in_func: TestFunction[_P, None],
        func_args: ArgDescr,
    ) -> None:
        func = in_func.deep_clone()
        task = lct.task()(func)
        self.check_valid_registered_task(task)
        # arguments correct, but we have an extra kwarg

        kwargs_excn_re = re.compile(
            r"Task does not have keyword argument\(s\): .*"
        )
        with pytest.raises(TypeError, match=kwargs_excn_re):
            task(
                *func_args.args(),
                this_argument_does_not_exist="Lorem ipsum dolor",
            )

        # doing the kwarg first makes no difference, still an error
        with pytest.raises(TypeError, match=kwargs_excn_re):
            task(
                this_argument_does_not_exist="Lorem ipsum dolor",
                *func_args.args(),
            )

        with pytest.raises(
            TypeError,
            match=r"Task expects (\d+) parameters, but (\d+) were passed",
        ):
            # We also test the "default value" scalar functions, and I am too
            # lazy to do a special-case test for them, so pass far too many
            # arguments so we cover them all.
            task(
                *func_args.args(),
                "This argument does not exist!",
                "This one doesn't either",
                "and neither does this one",
                "or this one",
                "or this one!!!",
                "Lorem",
                "ipsum",
                "dolor",
                "sit",
                "amet",
            )

    def test_decorator(self) -> None:
        @lct.task
        def foo() -> None:
            pass

        self.check_valid_registered_task(foo)
        foo()

    @pytest.mark.parametrize("register", [True, False])
    def test_decorator_kwargs(self, register: bool) -> None:
        @lct.task(register=register)
        def bar() -> None:
            pass

        if not register:
            self.check_valid_unregistered_task(bar)
            bar.complete_registration()

        self.check_valid_registered_task(bar)
        bar()

    # TODO(jfaibussowit)
    @pytest.mark.xfail
    def test_raised_exception(self) -> None:
        class CustomException(Exception):
            pass

        @lct.task
        def raises_exception() -> None:
            raise CustomException("There is no peace but the Pax Romana")

        with pytest.raises(
            CustomException, match=r"There is no peace but the Pax Romana"
        ):
            raises_exception()

    def test_constraint_kwargs(self) -> None:
        @lct.task
        def my_task(x: InputStore, y: InputStore, z: OutputStore) -> None:
            x_arr = np.asarray(x.get_inline_allocation())
            y_arr = np.asarray(y.get_inline_allocation())
            z_arr = np.asarray(z.get_inline_allocation())
            assert x_arr.shape == y_arr.shape
            assert z_arr.shape == y_arr.shape
            np.testing.assert_allclose(x_arr, 1)
            np.testing.assert_allclose(y_arr, 2)
            z_arr[:] = x_arr + y_arr

        x = make_input_store(value=1)
        y = make_input_store(value=2)
        z = make_output_store()
        my_task(x, y, z, task_constraints=(lg.align(x, y), lg.align(y, z)))
        get_legate_runtime().issue_execution_fence(block=True)
        np.testing.assert_allclose(
            np.asarray(z.get_physical_store().get_inline_allocation()),
            np.full(shape=tuple(x.shape), fill_value=3),
        )

    @pytest.mark.parametrize("shape", ((10,), (10, 10), (10, 10, 10)))
    def test_broadcast_constraint(self, shape: tuple[int, ...]) -> None:
        x_val = 456

        @lct.task
        def broadcast_task(x: InputStore, y: OutputStore) -> None:
            x_arr = np.asarray(x.get_inline_allocation())
            assert x_arr.shape == shape
            np.testing.assert_allclose(x_arr, x_val)
            y_arr = np.asarray(y.get_inline_allocation())
            assert y_arr.shape <= shape
            y_arr[:] = x_arr[tuple(map(slice, y_arr.shape))]

        x = make_input_store(value=x_val, shape=shape)
        y = make_output_store(shape=shape)
        broadcast_task(x, y, task_constraints=(lg.broadcast(x),))
        get_legate_runtime().issue_execution_fence(block=True)
        np.testing.assert_allclose(
            np.asarray(y.get_physical_store().get_inline_allocation()),
            np.full(shape=tuple(x.shape), fill_value=x_val),
        )

    @pytest.mark.parametrize("shape", ((1,), (10,), (100,)))
    @pytest.mark.parametrize("scaling_factor", (2, 3, 4))
    def test_scale_constraint(
        self, shape: tuple[int, ...], scaling_factor: int
    ) -> None:
        @lct.task
        def scale_task(x: InputStore, y: OutputStore) -> None:
            x_arr = np.asarray(x.get_inline_allocation())
            y_arr = np.asarray(y.get_inline_allocation())
            assert x_arr.shape == tuple(
                s * scaling_factor for s in y_arr.shape
            )
            # Compact x into y. The number of values compacted =
            # scaling_factor. For example, if scaling_factor = 2:
            #
            # x: [1, 1, 1, 1, ..., 1, 1]
            #     | /   | /        | /
            #     +     +          +
            # y: [2,    2,    ..., 2]
            y_arr[:] = x_arr.reshape((-1, scaling_factor)).sum(axis=-1)

        x = make_input_store(
            value=1, shape=tuple(s * scaling_factor for s in shape)
        )  # bigger
        y = make_output_store(shape=shape)  # smaller
        scale_task(x, y, task_constraints=(lg.scale((scaling_factor,), y, x),))
        get_legate_runtime().issue_execution_fence(block=True)
        np.testing.assert_allclose(
            np.asarray(y.get_physical_store().get_inline_allocation()),
            np.full(shape=tuple(y.shape), fill_value=scaling_factor),
        )


class TestVariantInvoker(BaseTest):
    @pytest.mark.parametrize("func", USER_FUNCS)
    def test_create_auto(self, func: TestFunction[_P, None]) -> None:
        invoker = VariantInvoker(func)

        self.check_valid_invoker(invoker, func)

    @pytest.mark.parametrize("func", UNTYPED_FUNCS)
    def test_untyped_funcs(self, func: TestFunction[_P, None]) -> None:
        with pytest.raises(TypeError):
            invoker = VariantInvoker(func)
            assert callable(invoker)  # unreachable

    def test_validate_good(self) -> None:
        def single_input_copy(a: InputStore) -> None:
            pass

        invoker = VariantInvoker(single_input)

        assert invoker.valid_signature(single_input_copy)
        invoker.validate_signature(single_input_copy)

    def test_validate_bad(self) -> None:
        invoker = VariantInvoker(single_input)

        assert not invoker.valid_signature(multi_output)

        with pytest.raises(ValueError):
            invoker.validate_signature(multi_output)

    @pytest.mark.parametrize(
        "func, func_args", zip(USER_FUNCS, USER_FUNC_ARGS)
    )
    def test_prepare_call_good(
        self,
        func: TestFunction[_P, None],
        func_args: ArgDescr,
        fake_auto_task: FakeAutoTask,
    ) -> None:
        invoker = VariantInvoker(func)
        args = func_args.inputs
        kwargs = dict()
        for attr in ("outputs", "scalars"):
            if attr_vals := getattr(func, attr, None):
                for name, val in zip(attr_vals, getattr(func_args, attr)):
                    kwargs[name] = val
        invoker.prepare_call(fake_auto_task, args, kwargs)

    def test_prepare_call_bad(self, fake_auto_task: FakeAutoTask) -> None:
        invoker = VariantInvoker(single_input)

        with pytest.raises(TypeError):
            # should be an InputStore, not int
            invoker.prepare_call(fake_auto_task, tuple(), {"a": 1})

        with pytest.raises(TypeError):
            # should be an InputStore, not int
            invoker.prepare_call(
                fake_auto_task,
                tuple(
                    1,  # type: ignore [arg-type]
                ),
                {},
            )

        input_store = make_input_store()
        with pytest.raises(ValueError):
            # duplicate values for 'a'
            invoker.prepare_call(
                fake_auto_task, (input_store,), {"a": input_store}
            )

    def test_prepare_call_wrong_arg_num(
        self, fake_auto_task: FakeAutoTask
    ) -> None:
        invoker = VariantInvoker(single_input)
        with pytest.raises(TypeError):
            # no arguments given
            invoker.prepare_call(fake_auto_task, tuple(), {})

        invoker = VariantInvoker(multi_input)
        with pytest.raises(TypeError):
            # no arguments given
            invoker.prepare_call(fake_auto_task, tuple(), {})

        with pytest.raises(TypeError):
            # missing 'a'
            invoker.prepare_call(
                fake_auto_task, tuple(), {"b": make_input_store()}
            )

        with pytest.raises(TypeError):
            # missing 'b'
            invoker.prepare_call(fake_auto_task, (make_input_store(),), {})

    @pytest.mark.parametrize(
        "in_func, func_args", zip(USER_FUNCS, USER_FUNC_ARGS)
    )
    @pytest.mark.parametrize("check_called", [True, False])
    def test_invoke_auto(
        self,
        in_func: TestFunction[_P, None],
        func_args: ArgDescr,
        check_called: bool,
    ) -> None:
        func = in_func.deep_clone()
        invoker = VariantInvoker(func)

        self.check_valid_invoker(invoker, func)

        ctx = FakeTaskContext()
        ctx.inputs = tuple(map(FakeArray, func_args.inputs))
        ctx.outputs = tuple(map(FakeArray, func_args.outputs))
        ctx.scalars = tuple(map(FakeScalar, func_args.scalars))

        assert not func.called
        invoker(ctx, func)
        if check_called:
            self.check_func_called(func)

    def test_reserved_args(self) -> None:
        def foo(task_constraints: InputStore) -> None:
            pass

        with pytest.raises(TypeError, match="Parameter name.* is not allowed"):
            VariantInvoker(foo)


class TestTaskUtil:
    @pytest.mark.parametrize("variant_kind", sorted(lct._util.KNOWN_VARIANTS))
    def test_validate_variant_good(self, variant_kind: str) -> None:
        assert isinstance(variant_kind, str)
        lct._util.validate_variant(variant_kind)  # type: ignore[arg-type]

    def test_validate_variant_bad(self) -> None:
        with pytest.raises(ValueError):
            lct._util.validate_variant(
                "this_variant_does_not_exist"  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    import sys

    # add -s to args, we do not want pytest to capture stdout here since this
    # gobbles any C++ exceptions
    sys.exit(pytest.main(sys.argv + ["-s"]))
