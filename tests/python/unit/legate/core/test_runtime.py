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

import json

import pytest

from legate.core import Scope, get_legate_runtime, track_provenance


@track_provenance()
def func() -> str:
    return Scope.provenance()


@track_provenance()
def unnested() -> str:
    return func()


@track_provenance(nested=True)
def nested() -> str:
    return func()


class Test_track_provenance:
    def test_unnested(self) -> None:
        human, machine = json.loads(unnested())
        assert "test_runtime.py" in human
        assert "test_runtime.py" in machine["file"]
        assert "line" in machine

    def test_nested(self) -> None:
        human, machine = json.loads(nested())
        assert "test_runtime.py" in human
        assert "test_runtime.py" in machine["file"]
        assert "line" in machine


class TestShutdownCallback:
    counter = 0

    @classmethod
    def increase(cls) -> None:
        cls.counter += 1

    @classmethod
    def reset(cls) -> None:
        cls.counter = 0

    @classmethod
    def assert_reset(cls) -> None:
        assert cls.counter == 0

    def test_basic_shutdown_callback(self) -> None:
        count = TestShutdownCallback.counter
        runtime = get_legate_runtime()
        runtime.add_shutdown_callback(TestShutdownCallback.increase)
        runtime.finish()
        # run it another time to check callback is not lingering
        runtime.finish()
        assert TestShutdownCallback.counter == count + 1

    def test_LIFO(self) -> None:
        TestShutdownCallback.counter = 99
        runtime = get_legate_runtime()
        runtime.add_shutdown_callback(TestShutdownCallback.increase)
        runtime.add_shutdown_callback(TestShutdownCallback.assert_reset)
        runtime.add_shutdown_callback(TestShutdownCallback.reset)
        runtime.finish()
        assert TestShutdownCallback.counter == 1

    def test_duplicate_callback(self) -> None:
        count = TestShutdownCallback.counter
        runtime = get_legate_runtime()
        for i in range(5):
            runtime.add_shutdown_callback(TestShutdownCallback.increase)
        runtime.finish()
        assert TestShutdownCallback.counter == count + 5


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
