# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from subprocess import CompletedProcess

import pytest

from legate.core import (
    Field,
    StoreTarget,
    Table,
    TaskTarget,
    get_legate_runtime,
    offload_to,
    types as ty,
)
from legate.core.data_interface import (
    MAX_DATA_INTERFACE_VERSION,
    LegateDataInterfaceItem,
    as_logical_array,
)

from .util.task_util import make_input_array


class Test_as_logical_array:
    def test_good(self) -> None:
        field = Field("foo", dtype=ty.int64)
        x = make_input_array(value=22)

        as_logical_array(Table([field], [x]))

    def test_identity(self) -> None:
        x = make_input_array(value=123)
        x2 = as_logical_array(x)

        # An array passed through as_logical_array() should just return itself.
        assert x2 is x

    def test_missing_interface(self) -> None:
        class MissingInterface:
            pass

        missing = MissingInterface()

        with pytest.raises(
            TypeError, match="object does not provide Legate data interface"
        ):
            as_logical_array(missing)  # type: ignore [arg-type]

    def test_missing_version(self) -> None:
        class MissingVersion:
            @property
            def __legate_data_interface__(self) -> Any:
                return {}

        missing = MissingVersion()

        with pytest.raises(
            TypeError, match="Legate data interface missing a version number"
        ):
            as_logical_array(missing)

    @pytest.mark.parametrize("v", ("junk", 1.2))
    def test_bad_version(self, v: Any) -> None:
        class BadVersion:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": v,
                    "data": {Field("foo", dtype=ty.int64): make_input_array()},
                }

        bad = BadVersion()

        with pytest.raises(
            TypeError,
            match="Legate data interface version expected an integer, got",
        ):
            as_logical_array(bad)

    @pytest.mark.parametrize("v", (0, -1))
    def test_bad_low_version(self, v: int) -> None:
        class LowVersion:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": v,
                    "data": {Field("foo", dtype=ty.int64): make_input_array()},
                }

        lo = LowVersion()

        with pytest.raises(
            TypeError, match=f"Legate data interface version {v} is below"
        ):
            as_logical_array(lo)

    def test_bad_high_version(self) -> None:
        v = MAX_DATA_INTERFACE_VERSION + 1

        class HighVersion:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": v,
                    "data": {Field("foo", dtype=ty.int64): make_input_array()},
                }

        hi = HighVersion()

        with pytest.raises(
            NotImplementedError,
            match=f"Unsupported Legate data interface version {v}",
        ):
            as_logical_array(hi)

    def test_bad_missing_fields(self) -> None:
        class MissingFields:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {"version": 1, "data": {}}

        missing = MissingFields()

        with pytest.raises(
            TypeError, match="Legate data object has no fields"
        ):
            as_logical_array(missing)

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

        too_many = TooManyFields()

        with pytest.raises(
            NotImplementedError,
            match="Legate data interface objects with more than one store are unsupported",  # noqa: E501
        ):
            as_logical_array(too_many)


class TestOffload:
    @pytest.mark.skipif(
        len(get_legate_runtime().machine.only(TaskTarget.GPU)) == 0,
        reason="test requires GPU",
    )
    def test_store_offload(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        if run_subprocess:
            run_subprocess(
                __file__,
                "TestOffload::test_store_offload",
                {
                    "LEGATE_AUTO_CONFIG": "0",
                    "LEGATE_CONFIG": "--gpus 1 --fbmem 1",
                },
            )
            return

        # big enough for one, but not enough for two
        shape = (300, 300)

        store = get_legate_runtime().create_store(ty.float64, shape)
        store.fill(1)
        # put store into FBMEM
        offload_to(store, target=StoreTarget.FBMEM)

        field = Field("foo", dtype=ty.int64)
        x = make_input_array(value=22, shape=shape)
        obj = Table([field], [x])
        # put store back into SYSMEM and let store2 in
        offload_to(store, target=StoreTarget.SYSMEM)
        offload_to(obj, target=StoreTarget.FBMEM)

    @pytest.mark.skipif(
        len(get_legate_runtime().machine.only(TaskTarget.GPU)) == 0,
        reason="test requires GPU",
    )
    def test_offload_no_duplicate(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        if run_subprocess:
            run_subprocess(
                __file__,
                "TestOffload::test_offload_no_duplicate",
                {
                    "LEGATE_AUTO_CONFIG": "0",
                    "LEGATE_CONFIG": "--gpus 1 --fbmem 1",
                },
            )
            return
        lg_arr = get_legate_runtime().create_array(ty.float64, (300, 300))
        lg_arr.fill(33)
        offload_to(lg_arr, target=StoreTarget.FBMEM)
        offload_to(lg_arr, target=StoreTarget.FBMEM)

    @pytest.mark.skipif(
        len(get_legate_runtime().machine.only(TaskTarget.GPU)) == 0,
        reason="test requires GPU",
    )
    def test_store_offload_overload(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        """To make sure offload is actually happening."""
        if run_subprocess:
            msg = "Failed to allocate.*bytes on memory.*(of kind GPU_FB_MEM)"
            with pytest.raises(RuntimeError, match=msg):
                run_subprocess(
                    __file__,
                    "TestOffload::test_store_offload_overload",
                    {
                        "LEGATE_AUTO_CONFIG": "0",
                        "LEGATE_CONFIG": "--gpus 1 --fbmem 1",
                    },
                )
            return

        # big enough for one, but not enough for two
        shape = (300, 300)
        store = get_legate_runtime().create_store(ty.float64, shape)
        store.fill(1)
        field = Field("foo", dtype=ty.int64)
        x = make_input_array(value=22, shape=shape)
        obj = Table([field], [x])

        offload_to(store, target=StoreTarget.FBMEM)
        # this should hit OOM and abort the proc
        offload_to(obj, target=StoreTarget.FBMEM)


if __name__ == "__main__":
    import sys

    # add -s to args, we do not want pytest to capture stdout here since this
    # gobbles any C++ exceptions
    sys.exit(pytest.main([*sys.argv, "-s"]))
