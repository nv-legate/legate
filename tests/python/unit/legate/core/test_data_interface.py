# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import pytest

from legate.core import Field, Table, types as ty
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

        nullable = NullableField()

        with pytest.raises(
            NotImplementedError,
            match="Argument: 'x' Legate data interface objects with nullable fields are unsupported",  # noqa: E501
        ):
            as_logical_array(nullable)

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

        nullable = NullableStore()

        with pytest.raises(
            NotImplementedError,
            match="Argument: 'x' Legate data interface objects with nullable stores are unsupported",  # noqa: E501
        ):
            as_logical_array(nullable)


if __name__ == "__main__":
    import sys

    # add -s to args, we do not want pytest to capture stdout here since this
    # gobbles any C++ exceptions
    sys.exit(pytest.main([*sys.argv, "-s"]))
