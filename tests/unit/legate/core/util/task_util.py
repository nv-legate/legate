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

import copy
import itertools
from collections.abc import Callable
from typing import (
    Any,
    Generic,
    Optional,
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
    get_legate_runtime,
    types as ty,
)
from legate.core.task import InputArray, InputStore, OutputArray, OutputStore

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
        inputs: tuple[LogicalStore | LogicalArray, ...] = tuple(),
        outputs: tuple[LogicalStore | LogicalArray, ...] = tuple(),
        scalars: tuple[Any, ...] = tuple(),
        arg_order: Optional[tuple[int, ...]] = None,
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


class FakeScalar(Generic[_T]):
    def __init__(self, value: _T):
        self._v = value

    def value(self) -> _T:
        return self._v


class FakeStore(PhysicalStore):
    def __init__(self, store: LogicalStore) -> None:
        # purposefully don't init super here
        self._store = store


class FakeArray(PhysicalArray):
    def __init__(self, handle: LogicalStore | LogicalArray) -> None:
        # purposefully don't init super here
        self._handle = handle

    def data(self) -> PhysicalStore:
        assert isinstance(self._handle, LogicalStore)
        return FakeStore(self._handle)


class FakeTaskContext:
    def __init__(self) -> None:
        self.inputs: tuple[PhysicalArray, ...] = (
            FakeArray(make_input_store()),
            FakeArray(make_input_store()),
        )
        self.outputs: tuple[PhysicalArray, ...] = (
            FakeArray(make_output_store()),
            FakeArray(make_output_store()),
        )
        self.reductions: tuple[PhysicalArray, ...] = ()
        self.scalars: tuple[Scalar, ...] = (  # type: ignore [assignment]
            FakeScalar(1),
            FakeScalar(2.0),
        )


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


class TestFunction(Protocol[_P, _T]):
    __test__: bool = False
    called: bool
    self: TestFunction[_P, _T]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    scalars: tuple[str, ...]
    deep_clone: Callable[[], TestFunction[_P, _T]]
    mark_called: Callable[[], None]

    def __call__(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        ...


def test_function(
    inputs: tuple[str, ...] = tuple(),
    outputs: tuple[str, ...] = tuple(),
    scalars: tuple[str, ...] = tuple(),
) -> Callable[[Callable[_P, _T]], TestFunction[_P, _T]]:
    def wrapper(fn: Callable[_P, _T]) -> TestFunction[_P, _T]:
        fn = TYPE_CAST(TestFunction[_P, _T], fn)
        fn.called = False
        fn.inputs = inputs
        fn.outputs = outputs
        fn.scalars = scalars

        def deep_clone() -> TestFunction[_P, _T]:
            # Must issue blocking sync since the tasks access function
            # variables. If there are multiple tasks in flight, these will
            # clobber one another and hence produce race conditions.
            #
            # Since you cannot get a reference to the current local function
            # object without going through the function name (which will be
            # shared by all threads), we have no choice but to hard-sync.
            get_legate_runtime().issue_execution_fence(block=True)
            fn_copy = copy.deepcopy(fn)
            fn_copy.called = False
            fn_copy.self = fn_copy

            def mark_called() -> None:
                # FIXME: This assertion doesn't hold if more than one Python
                # task gets launched.
                # assert not fn_copy.self.called, f"{fn}, {fn_copy.self}"
                fn_copy.self.called = True

            fn_copy.self.mark_called = mark_called
            return fn_copy

        fn.deep_clone = deep_clone
        return fn

    return wrapper


# so that pytest ignores this
test_function.__test__ = False  # type: ignore


@test_function()
def noargs() -> None:
    noargs.self.mark_called()


@test_function(inputs=("a",))
def single_input(a: InputStore) -> None:
    assert_isinstance(a, PhysicalStore)
    single_input.self.mark_called()


@test_function(inputs=("a", "b"))
def multi_input(a: InputStore, b: InputStore) -> None:
    assert_isinstance(a, PhysicalStore)
    assert_isinstance(b, PhysicalStore)
    multi_input.self.mark_called()


@test_function(outputs=("a",))
def single_output(a: OutputStore) -> None:
    assert_isinstance(a, PhysicalStore)
    single_output.self.mark_called()


@test_function(outputs=("a", "b"))
def multi_output(a: OutputStore, b: OutputStore) -> None:
    assert_isinstance(a, PhysicalStore)
    assert_isinstance(b, PhysicalStore)
    multi_output.self.mark_called()


@test_function(scalars=("a",))
def single_scalar(a: int) -> None:
    assert_isinstance(a, int)
    single_scalar.self.mark_called()


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
    multi_scalar.self.mark_called()


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
    mixed_args.self.mark_called()


@test_function(inputs=("a",))
def single_array(a: InputArray) -> None:
    assert_isinstance(a, PhysicalArray)
    single_array.self.mark_called()


@test_function(inputs=("a", "b"))
def multi_array(a: InputArray, b: InputArray) -> None:
    assert_isinstance(a, PhysicalArray)
    assert_isinstance(b, PhysicalArray)
    multi_array.self.mark_called()


@test_function(inputs=("a", "b"), outputs=("c", "d"))
def mixed_array_store(
    a: InputArray, b: InputStore, c: OutputArray, d: OutputStore
) -> None:
    assert_isinstance(a, PhysicalArray)
    assert_isinstance(b, PhysicalStore)
    assert_isinstance(c, PhysicalArray)
    assert_isinstance(d, PhysicalStore)
    mixed_array_store.self.mark_called()


USER_FUNCS = (
    noargs,
    single_input,
    multi_input,
    single_output,
    multi_output,
    single_scalar,
    multi_scalar,
    mixed_args,
    single_array,
    multi_array,
    mixed_array_store,
)

USER_FUNC_ARGS = (
    ArgDescr(),
    # stores
    ArgDescr(inputs=(make_input_store(),)),
    ArgDescr(inputs=(make_input_store(), make_input_store())),
    ArgDescr(outputs=(make_output_store(),)),
    ArgDescr(outputs=(make_output_store(), make_output_store())),
    ArgDescr(scalars=(1,)),
    ArgDescr(scalars=(1, 2.0, complex(1, 2), "Asdasdasdadsasdasd", True)),
    ArgDescr(
        inputs=(make_input_store(), make_input_store()),
        outputs=(make_output_store(), make_output_store()),
        scalars=(10, 12.0),
    ),
    # arrays
    ArgDescr(inputs=(make_input_array(),)),
    ArgDescr(inputs=(make_input_array(), make_input_array())),
    ArgDescr(
        inputs=(make_input_array(), make_input_store()),
        outputs=(make_output_array(), make_output_store()),
    ),
    ArgDescr(),
)


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
