# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import pytest

from legate.core.data_interface import (
    MAX_DATA_INTERFACE_VERSION,
    LegateDataInterfaceItem,
    as_logical_store,
)

from .util.task_util import make_input_store


class Test_as_logical_store:
    def test_good(self) -> None:
        class GoodVersion:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": 2,
                    "data": {"foo": make_input_store(value=22)},
                }

        as_logical_store(GoodVersion())

    def test_identity(self) -> None:
        x = make_input_store(value=123)
        x2 = as_logical_store(x)

        # A store passed through as_logical_store() should just return itself.
        assert x2 is x

    def test_missing_interface(self) -> None:
        class MissingInterface:
            pass

        missing = MissingInterface()

        with pytest.raises(
            TypeError, match="object does not provide Legate data interface"
        ):
            as_logical_store(missing)  # type: ignore [arg-type]

    def test_missing_version(self) -> None:
        class MissingVersion:
            @property
            def __legate_data_interface__(self) -> Any:
                return {}

        missing = MissingVersion()

        with pytest.raises(
            TypeError, match="Legate data interface missing a version number"
        ):
            as_logical_store(missing)

    @pytest.mark.parametrize("v", ("junk", 1.2))
    def test_bad_version(self, v: Any) -> None:
        class BadVersion:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {"version": v, "data": {"foo": make_input_store()}}

        bad = BadVersion()

        with pytest.raises(
            TypeError,
            match="Legate data interface version expected an integer, got",
        ):
            as_logical_store(bad)

    @pytest.mark.parametrize("v", (0, -1))
    def test_bad_low_version(self, v: int) -> None:
        class LowVersion:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {"version": v, "data": {"foo": make_input_store()}}

        lo = LowVersion()

        with pytest.raises(
            TypeError, match=f"Legate data interface version {v} is below"
        ):
            as_logical_store(lo)

    def test_bad_high_version(self) -> None:
        v = MAX_DATA_INTERFACE_VERSION + 1

        class HighVersion:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {"version": v, "data": {"foo": make_input_store()}}

        hi = HighVersion()

        with pytest.raises(
            NotImplementedError,
            match=f"Unsupported Legate data interface version {v}",
        ):
            as_logical_store(hi)

    def test_bad_missing_fields(self) -> None:
        class MissingFields:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {"version": 2, "data": {}}

        missing = MissingFields()

        with pytest.raises(
            TypeError, match="Legate data object has no fields"
        ):
            as_logical_store(missing)

    def test_bad_multiple_fields(self) -> None:
        class TooManyFields:
            @property
            def __legate_data_interface__(self) -> LegateDataInterfaceItem:
                return {
                    "version": 2,
                    "data": {
                        "foo": make_input_store(),
                        "bar": make_input_store(),
                    },
                }

        too_many = TooManyFields()

        with pytest.raises(
            NotImplementedError,
            match="Legate data interface objects with more than one store are unsupported",  # noqa: E501
        ):
            as_logical_store(too_many)


if __name__ == "__main__":
    import sys

    # add -s to args, we do not want pytest to capture stdout here since this
    # gobbles any C++ exceptions
    sys.exit(pytest.main([*sys.argv, "-s"]))
