# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    cast as TYPE_CAST,
)

from legate.core import (
    AutoTask,
    LogicalArray,
    LogicalStore,
    PhysicalArray,
    PhysicalStore,
    Scalar,
    TaskContext,
    get_legate_runtime,
    types as ty,
)

# These need to always be imported so that runtime type-checking works
from legate.core.task import (  # noqa: TC001
    InputArray,
    InputStore,
    OutputArray,
    OutputStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable


_T = TypeVar("_T")
_P = ParamSpec("_P")


def make_input_store(
    value: int = 123, shape: tuple[int, ...] | None = None
) -> LogicalStore:
    if shape is None:
        shape = (10,)
    runtime = get_legate_runtime()
    store = runtime.create_store(ty.int64, shape=shape)
    scalar = Scalar(value, ty.int64)
    runtime.issue_fill(store, scalar)
    return store


def make_output_store(shape: tuple[int, ...] | None = None) -> LogicalStore:
    return make_input_store(shape=shape)


def make_input_array(
    value: int = 123, shape: tuple[int, ...] | None = None
) -> LogicalArray:
    return LogicalArray.from_store(make_input_store(value=value, shape=shape))


def make_output_array(shape: tuple[int, ...] | None = None) -> LogicalArray:
    return make_input_array(shape=shape)


class ArgDescr:
    def __init__(
        self,
        inputs: tuple[LogicalStore | LogicalArray, ...] = (),
        outputs: tuple[LogicalStore | LogicalArray, ...] = (),
        scalars: tuple[Any, ...] = (),
        arg_order: tuple[int, ...] | None = None,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.scalars = scalars
        if arg_order is None:
            arg_order = tuple(
                range(sum(map(len, (self.inputs, self.outputs, self.scalars))))
            )
        assert len(arg_order) == len(set(arg_order))
        self.arg_order = arg_order

    def args(self) -> list[Any]:
        all_args = tuple(
            itertools.chain(self.inputs, self.outputs, self.scalars)
        )
        assert len(self.arg_order) == len(all_args)
        return [all_args[idx] for idx in self.arg_order]


class FakeScalar(Scalar, Generic[_T]):
    def __init__(self, value: _T):
        self._v = value

    def value(self) -> _T:
        return self._v


class FakeStore(PhysicalStore):
    def __init__(self, store: LogicalStore) -> None:
        # purposefully don't init super here
        self._store = store


class FakeArray(PhysicalArray):
    def __init__(
        self, handle: LogicalStore | LogicalArray, nullable: bool = False
    ) -> None:
        # purposefully don't init super here
        self._handle = handle
        self._nullable = nullable

    def data(self) -> PhysicalStore:
        assert isinstance(self._handle, LogicalStore)
        return FakeStore(self._handle)


class FakeTaskContext(TaskContext):
    def __init__(self) -> None:
        self._fake_inputs: tuple[PhysicalArray, ...] = (
            FakeArray(make_input_store()),
            FakeArray(make_input_store()),
        )
        self._fake_outputs: tuple[PhysicalArray, ...] = (
            FakeArray(make_output_store()),
            FakeArray(make_output_store()),
        )
        self._fake_reductions: tuple[PhysicalArray, ...] = ()
        self._fake_scalars: tuple[Scalar, ...] = (
            FakeScalar(1),
            FakeScalar(2.0),
        )

    @property
    def inputs(self) -> tuple[PhysicalArray, ...]:
        return self._fake_inputs

    @inputs.setter
    def inputs(self, value: tuple[PhysicalArray, ...]) -> None:
        self._fake_inputs = value

    @property
    def outputs(self) -> tuple[PhysicalArray, ...]:
        return self._fake_outputs

    @outputs.setter
    def outputs(self, value: tuple[PhysicalArray, ...]) -> None:
        self._fake_outputs = value

    @property
    def reductions(self) -> tuple[PhysicalArray, ...]:
        return self._fake_reductions

    @reductions.setter
    def reductions(self, value: tuple[PhysicalArray, ...]) -> None:
        self._fake_reductions = value

    @property
    def scalars(self) -> tuple[Scalar, ...]:
        return self._fake_scalars

    @scalars.setter
    def scalars(self, value: tuple[Scalar, ...]) -> None:
        self._fake_scalars = value


class FakeAutoTask(AutoTask):
    def __init__(self) -> None:
        # Do not init super on purpose
        pass

    def add_input(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> None:
        pass

    def add_output(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> None:
        pass

    def add_reduction(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> None:
        pass

    def add_scalar_arg(self, *args: Any, **kwargs: Any) -> None:
        pass


def assert_isinstance(arg: Any, py_type: type[_T]) -> None:
    assert isinstance(arg, py_type), f"Expected: {py_type}, got {type(arg)}"


_T_co = TypeVar("_T_co", covariant=True)


class TestFunction(Protocol[_P, _T_co]):
    __test__: bool = False
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    scalars: tuple[str, ...]

    def __call__(*args: _P.args, **kwargs: _P.kwargs) -> _T_co: ...


def test_function(
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    scalars: tuple[str, ...] = (),
) -> Callable[[Callable[_P, _T]], TestFunction[_P, _T]]:
    def wrapper(fn: Callable[_P, _T]) -> TestFunction[_P, _T]:
        fn_ret = TYPE_CAST(TestFunction[_P, _T], fn)
        fn_ret.__test__ = False
        fn_ret.inputs = inputs
        fn_ret.outputs = outputs
        fn_ret.scalars = scalars
        return fn_ret

    return wrapper


# so that pytest ignores this
test_function.__test__ = False  # type: ignore[attr-defined]


@test_function()
def noargs() -> None:
    pass


@test_function(inputs=("a",))
def single_input(a: InputStore) -> None:
    assert_isinstance(a, PhysicalStore)


@test_function(inputs=("a", "b"))
def multi_input(a: InputStore, b: InputStore) -> None:
    assert_isinstance(a, PhysicalStore)
    assert_isinstance(b, PhysicalStore)


@test_function(outputs=("a",))
def single_output(a: OutputStore) -> None:
    assert_isinstance(a, PhysicalStore)


@test_function(outputs=("a", "b"))
def multi_output(a: OutputStore, b: OutputStore) -> None:
    assert_isinstance(a, PhysicalStore)
    assert_isinstance(b, PhysicalStore)


@test_function(scalars=("a",))
def single_scalar(a: int) -> None:
    assert_isinstance(a, int)


@test_function(scalars=("a", "b", "c", "d", "e"))
def multi_scalar(a: int, b: float, c: complex, d: str, e: bool) -> None:
    assert_isinstance(a, int)
    assert_isinstance(b, float)
    assert_isinstance(c, complex)
    assert_isinstance(d, str)
    # to test the object is OK
    d.encode()
    assert_isinstance(e, bool)
    assert e is True or e is False


@test_function(inputs=("a", "b"), outputs=("c", "d"), scalars=("e", "f"))
def mixed_args(
    a: InputStore,
    b: InputStore,
    c: OutputStore,
    d: OutputStore,
    e: int,
    f: float,
) -> None:
    assert_isinstance(a, PhysicalStore)
    assert_isinstance(b, PhysicalStore)
    assert_isinstance(c, PhysicalStore)
    assert_isinstance(d, PhysicalStore)
    assert_isinstance(e, int)
    assert_isinstance(f, float)


@test_function(inputs=("a",))
def single_array(a: InputArray) -> None:
    assert_isinstance(a, PhysicalArray)


@test_function(inputs=("a", "b"))
def multi_array(a: InputArray, b: InputArray) -> None:
    assert_isinstance(a, PhysicalArray)
    assert_isinstance(b, PhysicalArray)


@test_function(inputs=("a", "b"), outputs=("c", "d"))
def mixed_array_store(
    a: InputArray, b: InputStore, c: OutputArray, d: OutputStore
) -> None:
    assert_isinstance(a, PhysicalArray)
    assert_isinstance(b, PhysicalStore)
    assert_isinstance(c, PhysicalArray)
    assert_isinstance(d, PhysicalStore)


_USER_FUNCS_WITH_ARGS = (
    (noargs, ArgDescr()),
    (single_input, ArgDescr(inputs=(make_input_store(),))),
    (multi_input, ArgDescr(inputs=(make_input_store(), make_input_store()))),
    (single_output, ArgDescr(outputs=(make_output_store(),))),
    (
        multi_output,
        ArgDescr(outputs=(make_output_store(), make_output_store())),
    ),
    (single_scalar, ArgDescr(scalars=(1,))),
    (
        multi_scalar,
        ArgDescr(scalars=(1, 2.0, complex(1, 2), "Asdasdasdadsasdasd", True)),
    ),
    (
        mixed_args,
        ArgDescr(
            inputs=(make_input_store(), make_input_store()),
            outputs=(make_output_store(), make_output_store()),
            scalars=(10, 12.0),
        ),
    ),
    (single_array, ArgDescr(inputs=(make_input_array(),))),
    (multi_array, ArgDescr(inputs=(make_input_array(), make_input_array()))),
    (
        mixed_array_store,
        ArgDescr(
            inputs=(make_input_array(), make_input_store()),
            outputs=(make_output_array(), make_output_store()),
        ),
    ),
)

USER_FUNCS, USER_FUNC_ARGS = tuple(zip(*_USER_FUNCS_WITH_ARGS, strict=True))


def non_none_return_type(a: InputStore, b: OutputStore, c: int) -> int:
    return 0


def untyped_non_none_return_type(  # type: ignore[no-untyped-def]
    a, b, c
) -> int:
    return 0


def untyped_single(a) -> None:  # type: ignore[no-untyped-def]
    pass


def untyped_multi(a, b) -> None:  # type: ignore[no-untyped-def]
    pass


def untyped_some(a, b: InputStore, c) -> None:  # type: ignore[no-untyped-def]
    pass


UNTYPED_FUNCS = (
    non_none_return_type,
    untyped_non_none_return_type,
    untyped_single,
    untyped_multi,
    untyped_some,
)
