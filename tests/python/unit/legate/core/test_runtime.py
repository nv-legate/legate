# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import json
import atexit
import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from subprocess import CompletedProcess
from pathlib import Path

import pytest

from legate import settings
from legate.core import (
    Library,
    Machine,
    Scope,
    TaskContext,
    get_legate_runtime,
    track_provenance,
    types as ty,
)
from legate.core._lib.runtime.runtime import (  # type: ignore[attr-defined]
    _LegateOutputStream,
    _Provenance,
)
from legate.core.task import InputStore, task


class Test_track_provenance:
    @pytest.fixture
    def patch_provenance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_Provenance, "config_value", lambda: True)

    @pytest.mark.parametrize("nested", (True, False, None))
    def test_docstring(self, nested: bool | None) -> None:
        kw = {} if nested is None else {"nested": nested}

        @track_provenance(**kw)
        def func() -> None:
            """A docstring."""

        assert func.__doc__ == "A docstring."

    @pytest.mark.parametrize("nested", (True, False, None))
    def test_noop_without_profiling(self, nested: bool | None) -> None:
        def func() -> None:
            pass

        kw = {} if nested is None else {"nested": nested}
        wrapped = track_provenance(**kw)(func)

        assert wrapped is func

    @pytest.mark.usefixtures("patch_provenance")
    def test_unnested(self) -> None:
        @track_provenance()
        def func() -> str:
            return Scope.provenance()

        @track_provenance()
        def unnested() -> str:
            return func()

        human, machine = json.loads(unnested())
        assert "test_runtime.py" in human
        assert "test_runtime.py" in machine["file"]
        assert "line" in machine

    @pytest.mark.usefixtures("patch_provenance")
    def test_nested(self) -> None:
        @track_provenance()
        def func() -> str:
            return Scope.provenance()

        @track_provenance(nested=True)
        def nested() -> str:
            return func()

        human, machine = json.loads(nested())
        assert "test_runtime.py" in human
        assert "test_runtime.py" in machine["file"]
        assert "line" in machine

    @pytest.mark.usefixtures("patch_provenance")
    def test_no_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(inspect, "currentframe", lambda: None)

        @track_provenance()
        def func() -> str:
            return Scope.provenance()

        human, machine = json.loads(func())
        assert human == "<unknown>"
        assert machine["file"] == "<unknown>"


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

    def test_basic_shutdown_callback(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        if run_subprocess:
            run_subprocess(
                __file__,
                "TestShutdownCallback::test_basic_shutdown_callback",
                {},
                check=True,
            )
            return

        count = TestShutdownCallback.counter
        runtime = get_legate_runtime()
        runtime.add_shutdown_callback(TestShutdownCallback.increase)
        runtime.finish()
        # run it another time to check callback is not lingering
        runtime.finish()
        assert TestShutdownCallback.counter == count + 1

    def test_LIFO(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        if run_subprocess:
            run_subprocess(
                __file__, "TestShutdownCallback and test_LIFO", {}, check=False
            )
            return
        TestShutdownCallback.counter = 99
        runtime = get_legate_runtime()
        runtime.add_shutdown_callback(TestShutdownCallback.increase)
        runtime.add_shutdown_callback(TestShutdownCallback.assert_reset)
        runtime.add_shutdown_callback(TestShutdownCallback.reset)
        runtime.finish()
        assert TestShutdownCallback.counter == 1

    def test_duplicate_callback(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        if run_subprocess:
            run_subprocess(
                __file__, "TestShutdownCallback::test_LIFO", {}, check=True
            )
            return
        count = TestShutdownCallback.counter
        runtime = get_legate_runtime()
        for _ in range(5):
            runtime.add_shutdown_callback(TestShutdownCallback.increase)
        runtime.finish()
        assert TestShutdownCallback.counter == count + 5

    def test_atexit(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        if run_subprocess:
            run_subprocess(
                __file__, "TestShutdownCallback::test_atexit", {}, check=True
            )
            return
        count = TestShutdownCallback.counter
        runtime = get_legate_runtime()
        runtime.add_shutdown_callback(TestShutdownCallback.increase)
        atexit._run_exitfuncs()
        msg = "Legate runtime cannot be started after legate::finish is called"
        with pytest.raises(RuntimeError, match=msg):
            get_legate_runtime()
        assert TestShutdownCallback.counter == count + 1


class TestRealmBacktrace:
    def test_realm_backtrace_faulthandler(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        assert run_subprocess, (
            "Runtime init completed with "
            "realm_backtrace and python faulthandler both enabled"
        )
        env = {"REALM_BACKTRACE": "1", "PYTHONFAULTHANDLER": "1"}
        msg = (
            "REALM_BACKTRACE and the Python fault handler are "
            "mutually exclusive and cannot both be enabled"
        )
        with pytest.raises(RuntimeError, match=msg):
            run_subprocess(
                __file__, "test_realm_backtrace_faulthandler", env, check=True
            )

    @pytest.mark.parametrize("val", ["0", "1"])
    def test_realm_backtrace_set(
        self,
        val: str,
        run_subprocess: Callable[..., CompletedProcess[Any]] | None,
    ) -> None:
        if run_subprocess:
            env = {"REALM_BACKTRACE": val}
            run_subprocess(
                __file__,
                f"TestRealmBacktrace::test_realm_backtrace_set[{val}]",
                env,
                faulthandler=False,
            )
            return
        backtrace = os.getenv("REALM_BACKTRACE")
        assert backtrace is not None
        get_legate_runtime().finish()
        assert os.getenv("REALM_BACKTRACE") == backtrace

    def test_realm_backtrace_unset(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        if run_subprocess:
            run_subprocess(
                __file__, "TestRealmBacktrace::test_realm_backtrace_unset", {}
            )
            return
        # by the time we got here, runtime should have set the env var
        assert os.getenv("REALM_BACKTRACE") is not None
        get_legate_runtime().finish()
        assert os.getenv("REALM_BACKTRACE") is None

    def test_realm_backtrace_invalid(
        self, run_subprocess: Callable[..., CompletedProcess[Any]] | None
    ) -> None:
        assert run_subprocess, (
            "Runtime init completed with invalid REALM_BACKTRACE value"
        )

        env = {"REALM_BACKTRACE": "foo"}
        with pytest.raises(
            RuntimeError, match="Invalid value for REALM_BACKTRACE"
        ):
            run_subprocess(
                __file__,
                "TestRealmBacktrace::test_realm_backtrace_invalid",
                env,
                faulthandler=False,
            )


class TestRuntime:
    def test_create_library(self) -> None:
        runtime = get_legate_runtime()
        runtime.create_library("foo")
        assert runtime.find_library("foo")

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

    @pytest.mark.skipif(
        not settings.settings.limit_stdout(),
        reason="test requires limit_stdout",
    )
    def test_limit_stdout(self, capsys: pytest.CaptureFixture[Any]) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(
            ty.int64, shape=(len(runtime.machine) * 10,)
        )
        store.fill(0)

        single = False

        @task
        def task_with_stdout(ctx: TaskContext, _: InputStore) -> None:
            nonlocal single
            single = ctx.is_single_task()
            sys.stdout.writelines("foo")

        task_with_stdout(store)
        runtime.issue_execution_fence(block=True)
        out, err = capsys.readouterr()
        expected = "foo" if single else "foo" * len(runtime.machine)
        assert out.strip() == expected
        sys.stdout.write("bar")
        out, err = capsys.readouterr()
        if runtime.node_id != 0:
            assert not out
        else:
            assert out == "bar"
        assert not err

    def test_output_stream(self, tmp_path: Path) -> None:
        runtime = get_legate_runtime()
        tmp_file = tmp_path / "tmp.txt"
        # not using context here to close through _LegateOutputStream later
        f = Path.open(tmp_file, "w")
        stream = _LegateOutputStream(f, runtime.node_id)
        assert stream.fileno() == f.fileno()
        assert not stream.isatty()
        stream.write("foo")
        stream.writelines("bar")
        stream.flush()
        stream.close()
        assert f.closed
        with pytest.raises(ValueError, match="I/O operation on closed file"):
            stream.fileno()
        with Path.open(tmp_file, "r") as f:
            content = f.readlines()
        assert "\n".join(content) == "foobar"
        # for code coverage
        with pytest.raises(AttributeError):
            stream.set_parent(stream)


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
