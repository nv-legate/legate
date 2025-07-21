# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import json
import atexit
from subprocess import PIPE, STDOUT, run

import pytest

from legate.core import (
    Library,
    Machine,
    Scope,
    TaskTarget,
    get_legate_runtime,
    track_provenance,
)


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
    @pytest.mark.parametrize("nested", (True, False, None))
    def test_docstring(self, nested: bool | None) -> None:
        kw = {} if nested is None else {"nested": nested}

        @track_provenance(**kw)
        def func() -> None:
            """A docstring."""

        assert func.__doc__ == "A docstring."

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


@pytest.mark.xfail(
    run=False, reason="should only be invoked by test_shutdown_callback()"
)
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
        for _ in range(5):
            runtime.add_shutdown_callback(TestShutdownCallback.increase)
        runtime.finish()
        assert TestShutdownCallback.counter == count + 5

    def test_atexit(self) -> None:
        count = TestShutdownCallback.counter
        runtime = get_legate_runtime()
        runtime.add_shutdown_callback(TestShutdownCallback.increase)
        atexit._run_exitfuncs()
        msg = "Legate runtime cannot be started after legate::finish is called"
        with pytest.raises(RuntimeError, match=msg):
            get_legate_runtime()
        assert TestShutdownCallback.counter == count + 1


@pytest.mark.xfail(
    get_legate_runtime().machine.preferred_target == TaskTarget.GPU,
    reason="aborts python subprocess",
)
@pytest.mark.parametrize(
    "test_case",
    [f for f in dir(TestShutdownCallback) if f.startswith("test")],
    ids=str,
)
def test_shutdown_callback(test_case: str) -> None:
    try:
        import coverage  # type: ignore[import-not-found]  # noqa: F401, PLC0415

        cov_args = ["coverage", "run", "-m"]
    except ModuleNotFoundError:
        cov_args = []

    pruned_env = os.environ.copy()

    del pruned_env["REALM_BACKTRACE"]

    proc = run(
        [
            sys.executable,
            "-m",
            *cov_args,
            "pytest",
            __file__,
            "-k",
            f"TestShutdownCallback and {test_case}",
            "--runxfail",
        ],
        stdout=PIPE,
        stderr=STDOUT,
        check=False,
        env=pruned_env,
    )
    assert not proc.returncode, proc.stdout


class TestRuntime:
    def test_create_library(self) -> None:
        runtime = get_legate_runtime()
        runtime.create_library("foo")
        assert runtime.find_library("foo")

    def test_get_new_task_id(self) -> None:
        runtime = get_legate_runtime()
        test_lib = runtime.create_library("test_get_new_task_id")
        # TODO(wonchanl) [issue 1435]
        # can't configure libraries in python
        # terminate called after throwing an instance of 'std::overflow_error'
        #   what():  The scope ran out of IDs
        with pytest.raises(OverflowError, match="The scope ran out of IDs"):
            test_lib.get_new_task_id()

    def test_submit_non_task(self) -> None:
        msg = "Unknown type of operation"
        runtime = get_legate_runtime()
        with pytest.raises(TypeError, match=msg):
            runtime.submit("foo")  # type: ignore[arg-type]

    def test_properties(self) -> None:
        runtime = get_legate_runtime()
        # just checking existence of these properties
        assert isinstance(runtime.node_id, int)
        assert isinstance(runtime.node_count, int)
        assert isinstance(runtime.machine, Machine)
        assert isinstance(runtime.core_library, Library)

    def test_issue_mapping_fence(self) -> None:
        # just checking nothing's broken by it
        get_legate_runtime().issue_mapping_fence()


class TestRuntimeError:
    def test_find_invalid_library(self) -> None:
        runtime = get_legate_runtime()
        msg = "Library test_find_invalid_library does not exist"
        with pytest.raises(ValueError, match=msg):
            runtime.find_library("test_find_invalid_library")

    def test_get_unconfigured_task_id(self) -> None:
        runtime = get_legate_runtime()
        test_lib = runtime.create_library("test_get_unconfigured_task_id")
        with pytest.raises(OverflowError, match="The scope ran out of IDs"):
            test_lib.get_new_task_id()


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
