# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
from dataclasses import dataclass

import pytest

import legate.utils.args as m

from ...util import Capsys


@dataclass(frozen=True)
class _TestObj:
    a: int = 10
    b: m.NotRequired[int] = m.Unset
    c: m.NotRequired[str] = "foo"
    d: m.NotRequired[str] = m.Unset


class TestArgSpec:
    def test_default(self) -> None:
        spec = m.ArgSpec("dest")
        assert spec.dest == "dest"
        assert spec.action == "store_true"

        # all others are unset
        assert set(m.entries(spec)) == {
            ("dest", "dest"),
            ("action", "store_true"),
        }


def test_entries() -> None:
    assert set(m.entries(_TestObj())) == {("a", 10), ("c", "foo")}


class Test_parse_library_command_args:
    @pytest.mark.parametrize("name", ("1foo", "a.b", "a/b", "a[", "a("))
    def test_bad_libname(self, name: str) -> None:
        with pytest.raises(ValueError):
            m.parse_library_command_args(name, [])

    def test_default_help(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Capsys
    ) -> None:
        monkeypatch.setattr("sys.argv", ["app", "-foo:help"])
        with pytest.raises(SystemExit) as e:
            m.parse_library_command_args("foo", [])
        assert e.value.code is None
        out, err = capsys.readouterr()  # type: ignore[unreachable]
        assert out.startswith("usage: <foo program>")

    def test_default_help_precedence(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Capsys
    ) -> None:
        monkeypatch.setattr("sys.argv", ["app", "-foo:help", "-foo:bar"])
        args = [m.Argument("bar", m.ArgSpec(dest="help"))]
        with pytest.raises(SystemExit) as e:
            m.parse_library_command_args("foo", args)
        assert e.value.code is None
        out, err = capsys.readouterr()  # type: ignore[unreachable]
        assert out.startswith("usage: <foo program>")

    def test_help_override(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Capsys
    ) -> None:
        monkeypatch.setattr("sys.argv", ["app", "-foo:help"])
        args = [m.Argument("help", m.ArgSpec(dest="help"))]
        ns = m.parse_library_command_args("foo", args)
        out, err = capsys.readouterr()
        assert out == ""
        assert vars(ns) == {"help": True}
        assert sys.argv == ["app"]

    def test_basic(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Capsys
    ) -> None:
        monkeypatch.setattr("sys.argv", ["app", "-foo:bar", "-foo:quux", "1"])
        args = [
            m.Argument("bar", m.ArgSpec(dest="bar")),
            m.Argument(
                "quux", m.ArgSpec(dest="quux", action="store", type=int)
            ),
        ]
        ns = m.parse_library_command_args("foo", args)
        out, err = capsys.readouterr()
        assert out == ""
        assert vars(ns) == {"bar": True, "quux": 1}
        assert sys.argv == ["app"]

    def test_extra_args_passed_on(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Capsys
    ) -> None:
        monkeypatch.setattr("sys.argv", ["app", "-foo:bar", "--extra", "1"])
        args = [m.Argument("bar", m.ArgSpec(dest="bar"))]
        ns = m.parse_library_command_args("foo", args)
        out, err = capsys.readouterr()
        assert out == ""
        assert vars(ns) == {"bar": True}
        assert sys.argv == ["app", "--extra", "1"]

    def test_unrecognized_libname_arg(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Capsys
    ) -> None:
        monkeypatch.setattr("sys.argv", ["app", "-foo:bar", "-foo:baz"])
        with pytest.warns(UserWarning) as record:
            ns = m.parse_library_command_args("foo", [])
        out, err = capsys.readouterr()
        assert out == ""
        assert vars(ns) == {}
        assert sys.argv == ["app", "-foo:bar", "-foo:baz"]

        # issues one warning for the first encountered
        assert len(record) == 1
        assert isinstance(record[0].message, Warning)
        assert (
            record[0].message.args[0]
            == "Unrecognized argument '-foo:bar' for foo (passed on as-is)"
        )
        assert out == ""
        assert vars(ns) == {}
        assert sys.argv == ["app", "-foo:bar", "-foo:baz"]


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
