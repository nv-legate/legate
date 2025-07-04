# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

import legate.util.settings as m

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


@contextmanager
def envset(
    value: Mapping[str, str] | None = None, **kw: Any
) -> Iterator[None]:
    old = os.environ.copy()
    if value:
        os.environ.update(value)
    os.environ.update(**kw)
    yield
    # take care to keep the same actual dict object
    os.environ.clear()
    os.environ.update(old)


class TestConverters:
    def test_convert_bool_1(self) -> None:
        assert m.convert_bool("1")

    def test_convert_bool_0(self) -> None:
        assert not m.convert_bool("0")

    @pytest.mark.parametrize("value", [True, False])
    def test_convert_bool_identity(self, value: bool) -> None:
        assert m.convert_bool(value) == value

    def test_convert_bool_bad(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            m.convert_bool("junk")


class TestPrioritizedSetting:
    def test_env_var_property(self) -> None:
        ps: Any = m.PrioritizedSetting("foo", env_var="LEGATE_FOO")
        assert ps.env_var == "LEGATE_FOO"

    def test_everything_unset_raises(self) -> None:
        ps: Any = m.PrioritizedSetting("foo")
        with pytest.raises(RuntimeError):
            ps()

    def test_implict_default(self) -> None:
        ps: Any = m.PrioritizedSetting("foo", default=10)
        assert ps() == 10

    def test_implict_default_converts(self) -> None:
        ps: Any = m.PrioritizedSetting("foo", convert=int, default="10")
        assert ps() == 10

    def test_help(self) -> None:
        ps: Any = m.PrioritizedSetting(
            "foo", env_var="LEGATE_FOO", default=10, help="bar"
        )
        assert ps.help == "bar"

    def test_name(self) -> None:
        ps: Any = m.PrioritizedSetting("foo", env_var="LEGATE_FOO", default=10)
        assert ps.name == "foo"

    def test_global_default(self) -> None:
        ps: Any = m.PrioritizedSetting("foo", env_var="LEGATE_FOO", default=10)
        assert ps.default == 10
        assert ps() == 10

    def test_local_default(self) -> None:
        ps: Any = m.PrioritizedSetting("foo", env_var="LEGATE_FOO", default=10)
        assert ps.default == 10
        assert ps(default=20) == 20

    def test_env_var(self) -> None:
        with envset(LEGATE_FOO="30"):
            ps: Any = m.PrioritizedSetting("foo", env_var="LEGATE_FOO")
            assert ps.env_var == "LEGATE_FOO"
            assert ps() == "30"
            assert ps(default=20) == "30"

    def test_env_var_converts(self) -> None:
        with envset(LEGATE_FOO="30"):
            ps: Any = m.PrioritizedSetting(
                "foo", convert=int, env_var="LEGATE_FOO"
            )
            assert ps() == 30

    def test_user_set(self) -> None:
        ps: Any = m.PrioritizedSetting("foo")
        ps.set_value(40)
        assert ps() == 40
        assert ps(default=20) == 40

    def test_user_unset(self) -> None:
        ps: Any = m.PrioritizedSetting("foo", default=2)
        ps.set_value(40)
        assert ps() == 40
        ps.unset_value()
        assert ps() == 2

    def test_user_set_converts(self) -> None:
        ps: Any = m.PrioritizedSetting("foo", convert=int)
        ps.set_value("40")
        assert ps() == 40

    def test_immediate(self) -> None:
        ps: Any = m.PrioritizedSetting("foo")
        assert ps(50) == 50
        assert ps(50, default=20) == 50

    def test_immediate_converts(self) -> None:
        ps: Any = m.PrioritizedSetting("foo", convert=int)
        assert ps("50") == 50

    def test_precedence(self) -> None:
        class FakeSettings:
            pass

        ps: Any = m.PrioritizedSetting(
            "foo", env_var="LEGATE_FOO", convert=int, default=0
        )

        # 0. global default
        assert ps() == 0

        # 1. local default
        assert ps(default=10) == 10

        # 2. environment variable
        with envset(LEGATE_FOO="40"):
            assert ps() == 40
            assert ps(default=10) == 40

            # 3. previously user-set value
            ps.set_value(60)
            assert ps() == 60
            assert ps(default=10) == 60

            # 4. immediate values
            assert ps(70) == 70
            assert ps(70, default=10) == 70

    def test_descriptors(self) -> None:
        class FakeSettings:
            foo: Any = m.PrioritizedSetting("foo", env_var="LEGATE_FOO")
            bar: Any = m.PrioritizedSetting(
                "bar", env_var="LEGATE_BAR", default=10
            )

        s = FakeSettings()
        assert s.foo is FakeSettings.foo

        assert s.bar() == 10
        s.bar = 20
        assert s.bar() == 20


class TestEnvOnlySetting:
    def test_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ps: Any = m.EnvOnlySetting(
            "foo", "FOO", default=10, convert=m.convert_int
        )
        monkeypatch.setenv("FOO", "5")
        assert ps() == 5

    def test_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ps: Any = m.EnvOnlySetting(
            "foo", "FOO", default=10, convert=m.convert_int
        )
        monkeypatch.delenv("FOO", raising=False)
        assert ps() == 10

    def test_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ps: Any = m.EnvOnlySetting(
            "foo", "FOO", default=10, convert=m.convert_int
        )
        monkeypatch.setenv("FOO", "5")
        assert ps() == 5
        monkeypatch.setenv("FOO", "6")
        assert ps() == 5

    def test_cache_using_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ps: Any = m.EnvOnlySetting(
            "foo", "FOO", default=10, convert=m.convert_int
        )
        monkeypatch.delenv("FOO", raising=False)
        assert ps() == 10
        monkeypatch.setenv("FOO", "6")
        assert ps() == 10

    def test_test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ps: Any = m.EnvOnlySetting(
            "foo", "FOO", default=10, test_default=11, convert=m.convert_int
        )
        monkeypatch.setenv("LEGATE_TEST", "1")
        assert ps() == 11

    def test_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ps: Any = m.EnvOnlySetting("foo", "FOO", convert=m.convert_int)
        monkeypatch.delenv("FOO", raising=False)
        with pytest.raises(ValueError, match="is not set and no default"):
            ps()
