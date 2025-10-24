# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import pytest

from legate.core import (
    StoreTarget,
    TaskTarget,
    get_legate_runtime,
    types as ty,
)

if TYPE_CHECKING:
    from legate.core import Type


# Python does not actually expose PyCapsule to python in any way, so this is
# the only way to check whether it is a capsule object
def is_capsule_type(obj: Any) -> bool:
    t = type(obj)
    return t.__module__ == "builtins" and t.__name__ == "PyCapsule"


def to_store_target(target: TaskTarget) -> StoreTarget:
    match target:
        case TaskTarget.CPU:
            return StoreTarget.SYSMEM
        case TaskTarget.OMP:
            return StoreTarget.SOCKETMEM
        case TaskTarget.GPU:
            return StoreTarget.FBMEM


class TestToDLPack:
    @pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 3, 3)))
    @pytest.mark.parametrize(
        "legate_type", (ty.int32, ty.float32, ty.uint64, ty.complex128)
    )
    @pytest.mark.parametrize("target", tuple(TaskTarget))
    def test_basic(
        self, shape: tuple[int, ...], legate_type: Type, target: TaskTarget
    ) -> None:
        runtime = get_legate_runtime()
        store_target = to_store_target(target)

        if runtime.machine.count(target) == 0:
            pytest.skip(
                f"Test requires support for {store_target} memory in order "
                "to be run"
            )

        store = runtime.create_store(dtype=legate_type, shape=shape)
        store.fill(3)

        phys = store.get_physical_store(target=store_target)
        assert phys.target == store_target
        capsule = phys.__dlpack__()
        assert is_capsule_type(capsule)
        # str(capsule) = '<capsule object "dltensor_versioned" at 0x103c97b50>'
        assert '"dltensor_versioned"' in str(capsule)

    @pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 3, 3)))
    @pytest.mark.parametrize(
        "legate_type", (ty.int32, ty.float32, ty.uint64, ty.complex128)
    )
    def test_to_numpy(self, shape: tuple[int, ...], legate_type: Type) -> None:
        VALUE = 3
        store = get_legate_runtime().create_store(
            dtype=legate_type, shape=shape
        )
        store.fill(VALUE)

        phys = store.get_physical_store()
        # Go through InlineAllocation so we ensure that asarray() isn't using
        # __dlpack__()
        alloc = phys.get_inline_allocation()
        assert not hasattr(alloc, "__dlpack__")
        assert (np.asarray(alloc) == VALUE).all()
        x = np.from_dlpack(phys)
        assert x.shape == store.shape
        assert x.dtype == store.type.to_numpy_dtype()
        assert x.size == store.size
        assert (x == VALUE).all()

    @pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 3, 3)))
    @pytest.mark.parametrize(
        "legate_type", (ty.int32, ty.float32, ty.uint64, ty.complex128)
    )
    def test_to_numpy_must_copy(
        self, shape: tuple[int, ...], legate_type: Type
    ) -> None:
        VALUE = 3
        store = get_legate_runtime().create_store(
            dtype=legate_type, shape=shape
        )
        store.fill(VALUE)

        phys = store.get_physical_store()
        x = np.from_dlpack(phys, copy=True)
        assert x.shape == store.shape
        assert x.dtype == store.type.to_numpy_dtype()
        assert x.size == store.size
        assert (x == VALUE).all()

        NEW_VALUE = 4
        x[:] = x.dtype.type(NEW_VALUE)  # type: ignore[unused-ignore, call-arg]

        assert (x == NEW_VALUE).all()

        phys = store.get_physical_store()
        # Go through InlineAllocation so we ensure that asarray() isn't using
        # __dlpack__()
        alloc = phys.get_inline_allocation()
        assert not hasattr(alloc, "__dlpack__")
        assert (np.asarray(alloc) == VALUE).all()

    @pytest.mark.parametrize("shape", ((1,), (2, 2), (3, 3, 3)))
    @pytest.mark.parametrize(
        "legate_type", (ty.int32, ty.float32, ty.uint64, ty.complex128)
    )
    def test_to_numpy_never_copy(
        self, shape: tuple[int, ...], legate_type: Type
    ) -> None:
        VALUE = 3
        store = get_legate_runtime().create_store(
            dtype=legate_type, shape=shape
        )
        store.fill(VALUE)

        phys = store.get_physical_store()
        x = np.from_dlpack(phys, copy=False)
        assert x.shape == store.shape
        assert x.dtype == store.type.to_numpy_dtype()
        assert x.size == store.size
        assert (x == VALUE).all()

        NEW_VALUE = 4
        x[:] = x.dtype.type(NEW_VALUE)  # type: ignore[unused-ignore, call-arg]

        assert (x == NEW_VALUE).all()

        phys = store.get_physical_store()
        # Go through InlineAllocation so we ensure that asarray() isn't using
        # __dlpack__()
        alloc = phys.get_inline_allocation()
        assert not hasattr(alloc, "__dlpack__")
        assert (np.asarray(alloc) == NEW_VALUE).all()

    @pytest.mark.skipif(
        TaskTarget.GPU not in get_legate_runtime().machine.valid_targets,
        reason="GPU only test",
    )
    @pytest.mark.xfail(reason="dlpack broken in FBMEM")
    @pytest.mark.parametrize("stream", [-1, 0, 2])
    def test_stream(self, stream: int) -> None:
        store = get_legate_runtime().create_store(
            dtype=ty.float32, shape=(3, 1, 3)
        )
        store.fill(3)
        phys = store.get_physical_store(target=StoreTarget.FBMEM)
        capsule = phys.__dlpack__(stream=stream)
        assert is_capsule_type(capsule)
        assert '"dltensor_versioned"' in str(capsule)


class TestToDLPackErrors:
    def test_invalid_dl_device(self) -> None:
        store = get_legate_runtime().create_store(
            dtype=ty.float32, shape=(3, 1, 3)
        )
        store.fill(3)

        phys = store.get_physical_store()
        with pytest.raises(BufferError, match=r"^$"):
            # There are legate RuntimeError messages dumped, but the actual
            # exception is BufferError which doesn't have any messages
            phys.__dlpack__(dl_device=(-1, 0))
