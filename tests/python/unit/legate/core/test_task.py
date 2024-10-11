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

import random
import re
from typing import Optional, ParamSpec, Union

import numpy as np
import pytest

import legate.core as lg
from legate.core import (
    PhysicalStore,
    Scalar,
    VariantCode,
    get_legate_runtime,
    task as lct,
    types as ty,
)
from legate.core.task import (
    ADD,
    InputArray,
    InputStore,
    OutputArray,
    OutputStore,
    PyTask,
    ReductionArray,
    ReductionStore,
    VariantInvoker,
)

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
    make_input_array,
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


class CustomException(Exception):
    pass


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

        kwargs_excn_re = re.compile(r"got an unexpected keyword argument '.*'")
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

        with pytest.raises(TypeError, match=r"too many positional arguments"):
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

    @pytest.mark.parametrize(
        "ExnType",
        (CustomException, ValueError, TypeError, RuntimeError, IndexError),
    )
    def test_raised_exception(self, ExnType: type) -> None:
        msg = "There is no peace but the Pax Romana"

        @lct.task(throws_exception=True)
        def raises_exception() -> None:
            raise ExnType(msg)

        with pytest.raises(ExnType, match=msg):
            raises_exception()
        auto_task = raises_exception.prepare_call()
        with pytest.raises(ExnType, match=msg):
            auto_task.execute()

    @pytest.mark.parametrize(
        "ExnType",
        (CustomException, ValueError, TypeError, RuntimeError, IndexError),
    )
    def test_deferred_exception(self, ExnType: type) -> None:
        @lct.task(throws_exception=True)
        def raises_exception() -> None:
            raise ExnType("There is no peace but the Pax Romana")

        with pytest.raises(
            ExnType, match=r"There is no peace but the Pax Romana"
        ):
            with lg.Scope(exception_mode=lg.ExceptionMode.DEFERRED):
                raises_exception()

    @pytest.mark.parametrize(
        "ExnType",
        (CustomException, ValueError, TypeError, RuntimeError, IndexError),
    )
    def test_ignored_exception(self, ExnType: type) -> None:
        @lct.task(throws_exception=True)
        def raises_exception() -> None:
            raise ExnType("There is no peace but the Pax Romana")

        with lg.Scope(exception_mode=lg.ExceptionMode.IGNORED):
            raises_exception()

    def test_legate_exception_handling(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64, (1,))

        @lct.task(constraints=(lg.broadcast("store", (1,)),))
        def task_func(store: OutputStore) -> None:
            assert False, "should never get here"

        msg = "Invalid broadcasting dimension"
        with pytest.raises(ValueError, match=msg):
            task_func(store)
        runtime.issue_execution_fence(block=True)

    def test_mixed_exception_declaration(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64, (1,))

        msg = "foo"

        @lct.task(throws_exception=False)
        def task_func(store: OutputStore) -> None:
            raise RuntimeError(msg)

        auto_task = task_func.prepare_call(store)
        auto_task.throws_exception(RuntimeError)
        with pytest.raises(RuntimeError, match=msg):
            runtime.submit(auto_task)

    def test_align_constraint(self) -> None:
        @lct.task(constraints=(lg.align("x", "y"), lg.align("y", "z")))
        def align_task(x: InputStore, y: InputStore, z: OutputStore) -> None:
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
        align_task(x, y, z)
        get_legate_runtime().issue_execution_fence(block=True)
        np.testing.assert_allclose(
            np.asarray(z.get_physical_store().get_inline_allocation()),
            np.full(shape=tuple(x.shape), fill_value=3),
        )

    @pytest.mark.parametrize("shape", ((10,), (10, 10), (10, 10, 10)))
    def test_broadcast_constraint(self, shape: tuple[int, ...]) -> None:
        x_val = 456

        @lct.task(constraints=(lg.broadcast("x"),))
        def broadcast_task(x: InputStore, y: OutputStore) -> None:
            x_arr = np.asarray(x.get_inline_allocation())
            assert x_arr.shape == shape
            np.testing.assert_allclose(x_arr, x_val)
            y_arr = np.asarray(y.get_inline_allocation())
            assert y_arr.shape <= shape
            y_arr[:] = x_arr[tuple(map(slice, y_arr.shape))]

        x = make_input_store(value=x_val, shape=shape)
        y = make_output_store(shape=shape)
        broadcast_task(x, y)
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
        @lct.task(constraints=(lg.scale((scaling_factor,), "y", "x"),))
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
        scale_task(x, y)
        get_legate_runtime().issue_execution_fence(block=True)
        np.testing.assert_allclose(
            np.asarray(y.get_physical_store().get_inline_allocation()),
            np.full(shape=tuple(y.shape), fill_value=scaling_factor),
        )

    def test_constraint_bad(self) -> None:
        def foo() -> None:
            pass

        x = 1
        with pytest.raises(
            TypeError,
            match=(
                "Constraint #1 of unexpected type. Found "
                f"{type(x)}, expected.*"
            ),
        ):
            lct.task(constraints=(x,))(foo)  # type: ignore[arg-type]

        def baz(x: OutputStore) -> None:
            pass

        with pytest.raises(
            ValueError,
            match=re.escape(
                r'Constraint argument "y" (of constraint "align()") '
                r"not in set of parameters: {'x'}"
            ),
        ):
            lct.task(constraints=(lg.align("x", "y"),))(baz)

    def test_reduction(self) -> None:
        @lct.task(throws_exception=True)
        def foo(store: ReductionStore[ADD]) -> None:
            assert_isinstance(store, PhysicalStore)

        x = make_input_store()
        foo(x)

    def test_scalar_arg_mismatching_dtype(self) -> None:
        @lct.task(throws_exception=True)
        def foo(scalar: Scalar) -> None:
            assert False, "should never get here"

        msg = "Task expected a value of type.*Scalar.*but got.*"
        with pytest.raises(TypeError, match=msg):
            foo(1.5)
        with pytest.raises(TypeError, match=msg):
            foo.prepare_call("foo")

    def test_default_arguments(self) -> None:
        x_val = 1
        y_val = 2

        @lct.task
        def task_with_default_args(
            x: InputStore,
            y: OutputStore,
            z_val: complex,
            z: complex = complex(1, 3),
        ) -> None:
            assert_isinstance(x, PhysicalStore)
            x_arr = np.asarray(x)
            np.testing.assert_allclose(
                x_arr, np.full(shape=tuple(x_arr.shape), fill_value=x_val)
            )

            assert_isinstance(y, PhysicalStore)
            y_arr = np.asarray(y)
            np.testing.assert_allclose(
                y_arr, np.full(shape=tuple(y_arr.shape), fill_value=y_val)
            )

            assert_isinstance(z_val, complex)
            assert_isinstance(z, complex)
            assert z == z_val

        x = make_input_store(value=x_val)
        y = make_input_store(value=y_val, shape=tuple(x.shape))
        # This line only exists to make the runtime actually issue the fill. y
        # is used as an output in the task (purely in order to test that
        # default values are properly ordered in the face of both inputs and
        # outputs), so normally the runtime would just skip the fill (because
        # the test is ostensibly writing to the task).
        #
        # Getting the physical store makes the runtime think we are about to
        # read the values, so it needs to materialize them.
        y.get_physical_store()
        task_with_default_args(x, y, complex(1, 3), complex(1, 3))
        get_legate_runtime().issue_execution_fence(block=True)

    def test_default_arguments_mixed(self) -> None:
        @lct.task
        def foo(x: int = 2, y: float = 4.5) -> None:
            assert_isinstance(x, int)
            assert x == 2
            assert_isinstance(y, float)
            assert y == 12.3

        # test that x isn't clobbered
        foo(y=12.3)
        get_legate_runtime().issue_execution_fence(block=True)

    def test_default_arguments_bad(self) -> None:
        # Have to do it like this because PhysicalStore() and PhysicalArray()
        # are not default-constructable (which is what default arguments
        # normally would be). But this test exist to make sure that some
        # determined user cannot get around this by constructing those default
        # objects elsewhere.
        phys_store = make_input_store().get_physical_store()
        phys_arr = make_input_array().get_physical_array()

        def foo(
            x: InputStore = phys_store,  # type: ignore[assignment]
        ) -> None:
            pass

        def foo1(x: InputArray = phys_arr) -> None:  # type: ignore[assignment]
            pass

        def foo2(
            x: OutputStore = phys_store,  # type: ignore[assignment]
        ) -> None:
            pass

        def foo3(
            x: OutputArray = phys_arr,  # type: ignore[assignment]
        ) -> None:
            pass

        functions = [foo, foo1, foo2, foo3]
        types = [InputStore, InputArray, OutputStore, OutputArray]
        assert len(functions) == len(types)
        for fn, store_ty in zip(functions, types):
            msg = re.escape(f"Default values for {store_ty} not yet supported")
            with pytest.raises(NotImplementedError, match=msg):
                lct.task(fn)  # type: ignore[call-overload]

    def test_default_reduction_arguments_bad(self) -> None:
        # Have to do it like this because PhysicalStore() and PhysicalArray()
        # are not default-constructable (which is what default arguments
        # normally would be). But this test exist to make sure that some
        # determined user cannot get around this by constructing those default
        # objects elsewhere.
        phys_store = make_input_store().get_physical_store()
        phys_arr = make_input_array().get_physical_array()

        def foo(
            x: ReductionStore[ADD] = phys_store,  # type: ignore[assignment]
        ) -> None:
            pass

        def foo1(
            x: ReductionArray[ADD] = phys_arr,  # type: ignore[assignment]
        ) -> None:
            pass

        functions = [foo, foo1]
        types = ["ReductionStore", "ReductionArray"]
        assert len(functions) == len(types)
        for fn, name in zip(functions, types):
            msg = re.escape(
                "Default values for "
                f"legate.core._ext.task.type.{name}[<ReductionOpKind.ADD: 0>] "
                "not yet supported"
            )
            with pytest.raises(NotImplementedError, match=msg):
                lct.task(fn)  # type: ignore[call-overload]

    def test_store_default_args(self) -> None:
        @lct.task
        def foo_or_none(x: InputStore | None = None) -> None:
            assert x is None

        foo_or_none()
        foo_or_none(None)

        @lct.task
        def foo_or_none_reversed(x: None | InputStore = None) -> None:
            assert x is None

        foo_or_none_reversed()
        foo_or_none_reversed(None)

        @lct.task
        def foo_optional(x: Optional[InputStore] = None) -> None:
            assert x is None

        foo_optional()
        foo_optional(None)

        @lct.task
        def foo_union(x: Union[InputStore, None] = None) -> None:
            assert x is None

        foo_union()
        foo_union(None)

    def test_array_default_args(self) -> None:
        @lct.task
        def foo_or_none(x: InputArray | None = None) -> None:
            assert x is None

        foo_or_none()
        foo_or_none(None)

        @lct.task
        def foo_or_none_reversed(x: None | InputArray = None) -> None:
            assert x is None

        foo_or_none_reversed()
        foo_or_none_reversed(None)

        @lct.task
        def foo_optional(x: Optional[InputArray] = None) -> None:
            assert x is None

        foo_optional()
        foo_optional(None)

        @lct.task
        def foo_union(x: Union[InputArray, None] = None) -> None:
            assert x is None

        foo_union()
        foo_union(None)


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
        with pytest.raises(
            TypeError, match=r"multiple values for argument 'a'"
        ):
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


class TestTaskUtil:
    @pytest.mark.parametrize("variant_kind", sorted(lct._util.KNOWN_VARIANTS))
    def test_validate_variant_good(self, variant_kind: VariantCode) -> None:
        assert isinstance(variant_kind, VariantCode)
        lct._util.validate_variant(variant_kind)

    def test_validate_variant_bad(self) -> None:
        with pytest.raises(ValueError):
            lct._util.validate_variant(345)  # type: ignore[arg-type]


if __name__ == "__main__":
    import sys

    # add -s to args, we do not want pytest to capture stdout here since this
    # gobbles any C++ exceptions
    sys.exit(pytest.main(sys.argv + ["-s"]))
