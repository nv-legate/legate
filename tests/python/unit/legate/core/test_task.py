# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import random
from typing import TYPE_CHECKING, Any, ParamSpec

import numpy as np

import pytest

import legate.core as lg
from legate.core import (
    Field,
    Library,
    LogicalArray,
    PhysicalStore,
    ResourceConfig,
    Scalar,
    Table,
    TaskContext,
    VariantCode,
    VariantOptions,
    get_legate_runtime,
    task as lct,
    types as ty,
)
from legate.core.data_interface import (
    MAX_DATA_INTERFACE_VERSION,
    LegateDataInterfaceItem,
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

if TYPE_CHECKING:
    from collections.abc import Generator

_P = ParamSpec("_P")


def repr_type_without_class(obj: type) -> str:
    # repr(SomeType) -> "<class 'legate.asdasd.SomeType'>"
    #
    # We just want "legate.asdasd.SomeType"
    return repr(obj).removeprefix("<class '").removesuffix("'>")


@pytest.fixture
def fake_auto_task() -> FakeAutoTask:
    return FakeAutoTask()


@pytest.fixture(autouse=True)
def auto_sync_runtime() -> Generator[None, None, None]:
    yield
    get_legate_runtime().issue_execution_fence(block=True)


class CustomException(Exception):
    pass


class BaseTest:
    def check_valid_task(self, task: PyTask) -> None:
        assert isinstance(task, PyTask)
        assert callable(task)
        assert isinstance(task.registered, bool)
        assert isinstance(task.library, Library)

    def check_valid_registered_task(self, task: PyTask) -> None:
        self.check_valid_task(task)
        assert task.registered
        task_id = task.task_id
        assert isinstance(task_id, int)
        assert task_id >= 0

    def check_valid_unregistered_task(self, task: PyTask) -> None:
        self.check_valid_task(task)
        assert not task.registered
        m = re.escape(
            "Task must complete registration "
            "(via task.complete_registration()) before receiving a task id"
        )
        with pytest.raises(RuntimeError, match=m):
            _ = task.task_id  # must complete registration first

    def check_valid_invoker(
        self, invoker: VariantInvoker, func: TestFunction[_P, None]
    ) -> None:
        assert callable(invoker)
        assert invoker.valid_signature(func)
        assert invoker.inputs == getattr(func, "inputs", ())
        assert invoker.outputs == getattr(func, "outputs", ())
        assert invoker.reductions == getattr(func, "reductions", ())
        assert invoker.scalars == getattr(func, "scalars", ())


class TestTask(BaseTest):
    def test_basic(self) -> None:
        def foo() -> None:
            pass

        variants = (VariantCode.CPU,)
        task = PyTask(func=foo, variants=variants)
        self.check_valid_registered_task(task)

    def test_construct_with_args(self) -> None:
        def foo(x: InputStore) -> None:
            pass

        variants = (VariantCode.CPU,)
        lib, _ = get_legate_runtime().find_or_create_library(
            "my_custom_library",
            config=ResourceConfig(max_tasks=10, max_dyn_tasks=10),
        )

        task = PyTask(
            func=foo,
            variants=variants,
            constraints=lg.broadcast("x"),
            options=VariantOptions(
                may_throw_exception=True, has_side_effect=True
            ),
            library=lib,
            register=False,
        )

        self.check_valid_unregistered_task(task)
        assert task.library == lib

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
        ("func", "func_args"), zip(USER_FUNCS, USER_FUNC_ARGS, strict=True)
    )
    @pytest.mark.parametrize("register", [True, False])
    def test_executable_auto(
        self, func: TestFunction[_P, None], func_args: ArgDescr, register: bool
    ) -> None:
        task = lct.task(register=register)(func)

        if not register:
            self.check_valid_unregistered_task(task)
            with pytest.raises(RuntimeError):
                task(*func_args.args())

            task.complete_registration()

        self.check_valid_registered_task(task)
        task(*func_args.args())

    @pytest.mark.parametrize(
        ("func", "func_args"), zip(USER_FUNCS, USER_FUNC_ARGS, strict=True)
    )
    @pytest.mark.parametrize("register", [True, False])
    def test_executable_prepare_call(
        self, func: TestFunction[_P, None], func_args: ArgDescr, register: bool
    ) -> None:
        task = lct.task(register=register)(func)

        if not register:
            self.check_valid_unregistered_task(task)
            with pytest.raises(RuntimeError):
                task(*func_args.args())

            task.complete_registration()

        self.check_valid_registered_task(task)
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

    def test_executable_wrong_arg_order(self) -> None:
        array_val = random.randint(0, 1000)
        c_val = random.randint(-1000, 1000)
        d_val = float(random.random() * c_val)

        def test_wrong_arg_order(
            a: InputStore, b: OutputStore, c: int, d: float
        ) -> None:
            assert_isinstance(a, PhysicalStore)
            assert np.asarray(a).all() == np.array([array_val] * 10).all()
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
        ("func", "func_args"), zip(USER_FUNCS, USER_FUNC_ARGS, strict=True)
    )
    def test_invoke_unhandled_args(
        self, func: TestFunction[_P, None], func_args: ArgDescr
    ) -> None:
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
                *func_args.args(),  # noqa: B026
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

        @lct.task(options=VariantOptions(may_throw_exception=True))
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
        @lct.task(options=VariantOptions(may_throw_exception=True))
        def raises_exception() -> None:
            msg = "There is no peace but the Pax Romana"
            raise ExnType(msg)

        with (
            pytest.raises(
                ExnType, match=r"There is no peace but the Pax Romana"
            ),
            lg.Scope(exception_mode=lg.ExceptionMode.DEFERRED),
        ):
            raises_exception()

    @pytest.mark.parametrize(
        "ExnType",
        (CustomException, ValueError, TypeError, RuntimeError, IndexError),
    )
    def test_ignored_exception(self, ExnType: type) -> None:
        @lct.task(options=VariantOptions(may_throw_exception=True))
        def raises_exception() -> None:
            msg = "There is no peace but the Pax Romana"
            raise ExnType(msg)

        with lg.Scope(exception_mode=lg.ExceptionMode.IGNORED):
            raises_exception()

    def test_legate_exception_handling(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64, (1,))

        @lct.task(constraints=(lg.broadcast("store", (1,)),))
        def task_func(store: OutputStore) -> None:
            pytest.fail("should never get here")

        msg = "Invalid broadcasting dimension"
        with pytest.raises(ValueError, match=msg):
            task_func(store)

    def test_mixed_exception_declaration(self) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(ty.int64, (1,))

        msg = "foo"

        @lct.task(options=VariantOptions(may_throw_exception=False))
        def task_func(store: OutputStore) -> None:
            raise RuntimeError(msg)

        auto_task = task_func.prepare_call(store)
        auto_task.throws_exception(RuntimeError)
        with pytest.raises(RuntimeError, match=msg):
            runtime.submit(auto_task)

    def test_align_constraint(self) -> None:
        @lct.task(constraints=(lg.align("x", "y"), lg.align("y", "z")))
        def align_task(x: InputStore, y: InputStore, z: OutputStore) -> None:
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            z_arr = np.asarray(z)
            assert x_arr.shape == y_arr.shape
            assert z_arr.shape == y_arr.shape
            np.testing.assert_allclose(x_arr, 1)
            np.testing.assert_allclose(y_arr, 2)
            z_arr[:] = x_arr + y_arr

        x = make_input_store(value=1)
        y = make_input_store(value=2)
        z = make_output_store()
        align_task(x, y, z)
        np.testing.assert_allclose(
            np.asarray(z.get_physical_store()),
            np.full(shape=tuple(x.shape), fill_value=3),
        )

    @pytest.mark.parametrize("shape", ((10,), (10, 10), (10, 10, 10)))
    def test_broadcast_constraint(self, shape: tuple[int, ...]) -> None:
        x_val = 456

        @lct.task(constraints=(lg.broadcast("x"),))
        def broadcast_task(x: InputStore, y: OutputStore) -> None:
            x_arr = np.asarray(x)
            assert x_arr.shape == shape
            np.testing.assert_allclose(x_arr, x_val)
            y_arr = np.asarray(y)
            assert y_arr.shape <= shape
            y_arr[:] = x_arr[tuple(map(slice, y_arr.shape))]

        x = make_input_store(value=x_val, shape=shape)
        y = make_output_store(shape=shape)
        broadcast_task(x, y)
        np.testing.assert_allclose(
            np.asarray(y.get_physical_store()),
            np.full(shape=tuple(x.shape), fill_value=x_val),
        )

    @pytest.mark.parametrize("shape", ((1,), (10,), (100,)))
    @pytest.mark.parametrize("scaling_factor", (2, 3, 4))
    def test_scale_constraint(
        self, shape: tuple[int, ...], scaling_factor: int
    ) -> None:
        @lct.task(constraints=(lg.scale((scaling_factor,), "y", "x"),))
        def scale_task(x: InputStore, y: OutputStore) -> None:
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
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
        np.testing.assert_allclose(
            np.asarray(y.get_physical_store()),
            np.full(shape=tuple(y.shape), fill_value=scaling_factor),
        )

    def test_constraint_bad(self) -> None:
        def foo() -> None:
            pass

        x = 1
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"'{repr_type_without_class(type(x))}' object is not iterable"
            ),
        ):
            lct.task(constraints=(x,))(foo)  # type: ignore[arg-type]

        def baz(x: OutputStore) -> None:
            pass

        m = re.escape(
            "constraint argument \"y\" not in set of parameters: {'x'}"
        )
        with pytest.raises(ValueError, match=m):
            lct.task(constraints=(lg.align("x", "y"),))(baz)

    def test_reduction(self) -> None:
        @lct.task(options=VariantOptions(may_throw_exception=True))
        def foo(store: ReductionStore[ADD]) -> None:
            assert_isinstance(store, PhysicalStore)

        x = make_input_store()
        foo(x)

    def test_scalar_arg_mismatching_dtype(self) -> None:
        @lct.task(options=VariantOptions(may_throw_exception=True))
        def foo(scalar: Scalar) -> None:
            pytest.fail("should never get here")

        msg = "Task expected a value of type.*Scalar.*but got.*"
        with pytest.raises(TypeError, match=msg):
            foo(1.5)
        with pytest.raises(TypeError, match=msg):
            foo.prepare_call("foo")

    def test_default_arguments(self) -> None:
        x_val = 1
        y_val = 2
        z_arg_value = complex(1, 3)

        @lct.task
        def task_with_default_args(
            x: InputStore,
            y: InputStore,
            z_val: complex,
            # Can be anything so long as it isn't the same as what we pass to
            # as the task arguments
            z: complex = complex(2, 6),
        ) -> None:
            assert_isinstance(x, PhysicalStore)
            x_arr = np.asarray(x)
            x_expected = np.full(shape=x_arr.shape, fill_value=x_val)
            np.testing.assert_allclose(x_arr, x_expected)

            assert_isinstance(y, PhysicalStore)
            y_arr = np.asarray(y)
            y_expected = np.full(shape=y_arr.shape, fill_value=y_val)
            np.testing.assert_allclose(y_arr, y_expected)

            assert_isinstance(z_val, complex)
            assert_isinstance(z, complex)
            assert z == z_arg_value
            assert z_val == z_arg_value
            assert z != complex(2, 6)  # The default value

        x = make_input_store(value=x_val)
        y = make_input_store(value=y_val, shape=tuple(x.shape))
        task_with_default_args(x, y, z_arg_value, z_arg_value)

    def test_default_arguments_mixed(self) -> None:
        @lct.task
        def foo(x: int = 2, y: float = 4.5) -> None:
            assert_isinstance(x, int)
            assert x == 2
            assert_isinstance(y, float)
            assert y == 12.3

        # test that x isn't clobbered
        foo(y=12.3)

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
        for fn, store_ty in zip(functions, types, strict=True):
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
        for fn, name in zip(functions, types, strict=True):
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
        def foo_optional(x: InputStore | None = None) -> None:
            assert x is None

        foo_optional()
        foo_optional(None)

        @lct.task
        def foo_union(x: InputStore | None = None) -> None:
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
        def foo_optional(x: InputArray | None = None) -> None:
            assert x is None

        foo_optional()
        foo_optional(None)

        @lct.task
        def foo_union(x: InputArray | None = None) -> None:
            assert x is None

        foo_union()
        foo_union(None)


class TestLegateDataInterface:
    def test_good(self) -> None:
        @lct.task
        def foo(x: InputArray) -> None:
            arr = np.asarray(x)
            assert arr.dtype == np.int64
            assert (arr == 22).all()

        field = Field("foo", dtype=ty.int64)
        x = make_input_array(value=22)

        foo(Table([field], [x]))

    def test_missing_version(self) -> None:
        class MissingVersion:
            @property
            def __legate_data_interface__(self) -> Any:
                return {}

        @lct.task
        def foo(x: InputArray) -> None:
            pytest.fail("Must never reach this point")

        missing = MissingVersion()

        with pytest.raises(
            TypeError,
            match="Argument: 'x' Legate data interface missing a version number",  # noqa: E501
        ):
            foo(missing)

    @pytest.mark.parametrize("v", ("junk", 1.2))
    def test_bad_version(self, v: Any) -> None:
        class BadVersion:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": v,
                    "data": {Field("foo", dtype=ty.int64): make_input_array()},
                }

        @lct.task
        def foo(x: InputArray) -> None:
            pytest.fail("Must never reach this point")

        bad = BadVersion()

        with pytest.raises(
            TypeError,
            match="Argument: 'x' Legate data interface version expected an integer, got",  # noqa: E501
        ):
            foo(bad)

    @pytest.mark.parametrize("v", (0, -1))
    def test_bad_low_version(self, v: int) -> None:
        class LowVersion:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": v,
                    "data": {Field("foo", dtype=ty.int64): make_input_array()},
                }

        @lct.task
        def foo(x: InputArray) -> None:
            pytest.fail("Must never reach this point")

        lo = LowVersion()

        with pytest.raises(
            TypeError,
            match=f"Argument: 'x' Legate data interface version {v} is below",
        ):
            foo(lo)

    def test_bad_high_version(self) -> None:
        v = MAX_DATA_INTERFACE_VERSION + 1

        class HighVersion:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": v,
                    "data": {Field("foo", dtype=ty.int64): make_input_array()},
                }

        @lct.task
        def foo(x: InputArray) -> None:
            pytest.fail("Must never reach this point")

        hi = HighVersion()

        with pytest.raises(
            NotImplementedError,
            match=f"Argument: 'x' Unsupported Legate data interface version {v}",  # noqa: E501
        ):
            foo(hi)

    def test_bad_missing_fields(self) -> None:
        class MissingFields:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {"version": 1, "data": {}}

        @lct.task
        def foo(x: InputArray) -> None:
            pytest.fail("Must never reach this point")

        missing = MissingFields()

        with pytest.raises(
            TypeError, match="Argument: 'x' Legate data object has no fields"
        ):
            foo(missing)

    def test_bad_multiple_fields(self) -> None:
        class TooManyFields:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": 1,
                    "data": {
                        Field("foo", dtype=ty.int64): make_input_array(),
                        Field("bar", dtype=ty.int64): make_input_array(),
                    },
                }

        @lct.task
        def foo(x: InputArray) -> None:
            pytest.fail("Must never reach this point")

        too_many = TooManyFields()

        with pytest.raises(
            NotImplementedError,
            match="Argument: 'x' Legate data interface objects with more than one store are unsupported",  # noqa: E501
        ):
            foo(too_many)

    # Can't currently even create a nullable field to test with
    @pytest.mark.xfail
    def test_bad_nullable_fields(self) -> None:
        class NullableField:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": 1,
                    "data": {
                        Field(
                            "foo", nullable=True, dtype=ty.int64
                        ): make_input_array()
                    },
                }

        @lct.task
        def foo(x: InputArray) -> None:
            pytest.fail("Must never reach this point")

        nullable = NullableField()

        with pytest.raises(
            NotImplementedError,
            match="Argument: 'x' Legate data interface objects with nullable fields are unsupported",  # noqa: E501
        ):
            foo(nullable)

    # Trying to create a nullable array, even a fake one, explodes
    @pytest.mark.skip
    def test_bad_nullable_array(self) -> None:
        class NullableStore:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": 1,
                    "data": {Field("foo", dtype=ty.int64): make_input_array()},
                }

        @lct.task
        def foo(x: InputArray) -> None:
            pytest.fail("Must never reach this point")

        nullable = NullableStore()

        with pytest.raises(
            NotImplementedError,
            match="Argument: 'x' Legate data interface objects with nullable stores are unsupported",  # noqa: E501
        ):
            foo(nullable)

    def test_task_properties(self) -> None:
        @lct.task
        def foo() -> None:
            pass

        # without this mypy thinks len(task.exception_types) == 0 after the
        # first assert, and complains the last line is unreachable.
        def safe_assert(expr: bool) -> None:
            assert expr

        task = foo.prepare_call()
        # just touching raw_handle for coverage
        _ = task.raw_handle
        safe_assert(len(task.exception_types) == 0)
        task.throws_exception(RuntimeError)
        safe_assert(len(task.exception_types) == 1)
        assert RuntimeError in task.exception_types


class TestVariantInvoker(BaseTest):
    @pytest.mark.parametrize("func", USER_FUNCS)
    def test_create_auto(self, func: TestFunction[_P, None]) -> None:
        invoker = VariantInvoker(func)

        self.check_valid_invoker(invoker, func)

    @pytest.mark.parametrize("func", UNTYPED_FUNCS)
    def test_untyped_funcs(self, func: TestFunction[_P, None]) -> None:
        with pytest.raises(TypeError):
            _ = VariantInvoker(func)

    def test_validate_good(self) -> None:
        def single_input_copy(a: InputStore) -> None:
            pass

        invoker = VariantInvoker(single_input)

        assert invoker.valid_signature(single_input_copy)
        invoker.validate_signature(single_input_copy)

    def test_validate_bad(self) -> None:
        invoker = VariantInvoker(single_input)

        assert not invoker.valid_signature(multi_output)

        with pytest.raises(ValueError):  # noqa: PT011
            invoker.validate_signature(multi_output)

    @pytest.mark.parametrize(
        ("func", "func_args"), zip(USER_FUNCS, USER_FUNC_ARGS, strict=True)
    )
    def test_prepare_call_good(
        self,
        func: TestFunction[_P, None],
        func_args: ArgDescr,
        fake_auto_task: FakeAutoTask,
    ) -> None:
        invoker = VariantInvoker(func)
        args = func_args.inputs
        kwargs = {}
        for attr in ("outputs", "scalars"):
            if attr_vals := getattr(func, attr, None):
                for name, val in zip(
                    attr_vals, getattr(func_args, attr), strict=True
                ):
                    kwargs[name] = val  # noqa: PERF403
        invoker.prepare_call(fake_auto_task, args, kwargs)

    def test_wrong_arg_num(self) -> None:
        func = single_input
        invoker = VariantInvoker(func)
        ctx = FakeTaskContext()
        ctx.inputs = tuple(
            map(FakeArray, (make_input_store(), make_input_store()))
        )
        msg = re.escape("Wrong number of given arguments (2), expected 1")
        with pytest.raises(ValueError, match=msg):
            invoker(ctx, func)

    def test_args_with_default_val(self) -> None:
        def args_default_val(
            a: TaskContext,
            b: InputStore | None = None,
            c: OutputStore | None = None,
        ) -> None:
            pass

        func = args_default_val
        invoker = VariantInvoker(func)
        ctx = FakeTaskContext()
        arr1 = make_input_array()
        arr2 = get_legate_runtime().create_array(ty.null_type, (1,))
        ctx.inputs = (arr1.get_physical_array(),)
        ctx.outputs = (arr2.get_physical_array(),)
        ctx.scalars = ()
        invoker(ctx, func)

    def test_prepare_call_constraints(self) -> None:
        runtime = get_legate_runtime()
        task = lct.task()(single_input)
        auto_task = runtime.create_auto_task(
            runtime.core_library, task.task_id
        )
        invoker = VariantInvoker(single_input)

        arg = make_input_store()
        variable = auto_task.declare_partition()
        constraints = lg.broadcast(variable)
        invoker.prepare_call(
            auto_task, (arg,), kwargs={}, constraints=tuple(constraints)
        )

    def test_prepare_call_bad(self, fake_auto_task: FakeAutoTask) -> None:
        invoker = VariantInvoker(single_input)

        with pytest.raises(TypeError):
            # should be an InputStore, not int
            invoker.prepare_call(fake_auto_task, (), {"a": 1})

        with pytest.raises(TypeError):
            # should be an InputStore, not int
            invoker.prepare_call(
                fake_auto_task,
                tuple(
                    1  # type: ignore [arg-type]
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
            invoker.prepare_call(fake_auto_task, (), {})

        invoker = VariantInvoker(multi_input)
        with pytest.raises(TypeError):
            # no arguments given
            invoker.prepare_call(fake_auto_task, (), {})

        with pytest.raises(TypeError):
            # missing 'a'
            invoker.prepare_call(fake_auto_task, (), {"b": make_input_store()})

        with pytest.raises(TypeError):
            # missing 'b'
            invoker.prepare_call(fake_auto_task, (make_input_store(),), {})

    @pytest.mark.parametrize(
        ("func", "func_args"), zip(USER_FUNCS, USER_FUNC_ARGS, strict=True)
    )
    def test_invoke_auto(
        self, func: TestFunction[_P, None], func_args: ArgDescr
    ) -> None:
        invoker = VariantInvoker(func)

        self.check_valid_invoker(invoker, func)

        ctx = FakeTaskContext()
        ctx.inputs = tuple(map(FakeArray, func_args.inputs))
        ctx.outputs = tuple(map(FakeArray, func_args.outputs))
        ctx.scalars = tuple(map(FakeScalar, func_args.scalars))

        invoker(ctx, func)


class TestTaskUtil:
    @pytest.mark.parametrize("variant_kind", tuple(VariantCode))
    def test_validate_variant_good(self, variant_kind: VariantCode) -> None:
        assert isinstance(variant_kind, VariantCode)
        lct._util.validate_variant(variant_kind)

    def test_validate_variant_bad(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            lct._util.validate_variant(345)  # type: ignore[arg-type]


class TestUnbound:
    @pytest.mark.parametrize("shape", ((1,), (1, 2), (1, 2, 3)), ids=str)
    def test_unbound_store(self, shape: tuple[int, ...]) -> None:
        VALUE = 123

        @lct.task(options=VariantOptions(has_allocations=True))
        def foo(x: OutputArray) -> None:
            buf = x.data().create_output_buffer(shape)
            np.asarray(buf)[:] = VALUE

        runtime = get_legate_runtime()
        x = runtime.create_array(ty.int64, ndim=len(shape))

        # Mypy considers boolean attributes to be immutable unless they are
        # directly assigned. I.e. given:
        #
        # assert x.bool_attr
        # foo(x)  # assume foo() modifies bool_attr
        # assert not x.bool_attr
        # baz()
        #
        # It believes that x.bool_attr holds the same value before and after
        # foo(). So it complains that the later call to baz() is unreachable
        # because *obviously* bool_attr cannot have changed.
        #
        # This issue has been raised previously
        # (https://github.com/python/mypy/issues/11969), but mypy seemingly
        # considers this a feature, not a bug.
        #
        # So we need this extra function to obfuscate the attribute access...
        def get_unbound(array: LogicalArray) -> bool:
            return array.unbound

        assert get_unbound(x)
        foo(x)
        runtime.issue_execution_fence(block=True)
        assert not get_unbound(x)
        assert x.type == ty.int64
        assert x.shape == shape

        x_phys = x.get_physical_array()
        x_arr = np.asarray(x_phys)
        assert (x_arr == VALUE).all()


if __name__ == "__main__":
    import sys

    # add -s to args, we do not want pytest to capture stdout here since this
    # gobbles any C++ exceptions
    sys.exit(pytest.main([*sys.argv, "-s"]))
