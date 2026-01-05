# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import platform
from typing import TYPE_CHECKING

import pytest

import legate.util.system as m
from legate.util.types import CPUInfo

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test___all__() -> None:
    assert m.__all__ == ("System",)


class TestSystem:
    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("____TEST", "10")

        s = m.System()

        expected = dict(os.environ)
        expected.update({"____TEST": "10"})
        assert s.env == expected

        assert id(s.env) != id(os.environ)

    @pytest.mark.parametrize("os", ("Linux", "Darwin"))
    def test_os_good(self, mocker: MockerFixture, os: str) -> None:
        mocker.patch("platform.system", return_value=os)

        s = m.System()

        assert s.os == os

    def test_os_bad(self, mocker: MockerFixture) -> None:
        mocker.patch("platform.system", return_value="junk")

        s = m.System()

        msg = "Legate does not work on junk"
        with pytest.raises(RuntimeError, match=msg):
            _ = s.os

    # These properties delegate to util functions, just verify plumbing

    def test_legate_paths(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "legate.util.system.get_legate_paths", return_value="legate paths"
        )

        s = m.System()

        assert s.legate_paths == "legate paths"  # type: ignore[comparison-overlap]

    def test_cpus(self) -> None:
        s = m.System()
        cpus = s.cpus
        assert len(cpus) > 0
        assert all(len(cpu.ids) > 0 for cpu in cpus)

    @pytest.mark.skipif(platform.system() != "Darwin", reason="OSX test")
    def test_gpus_osx(self) -> None:
        s = m.System()

        msg = "GPU execution is not available on OSX."
        with pytest.raises(RuntimeError, match=msg):
            _ = s.gpus


class Test_expand_range:
    def test_errors(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            m.expand_range("foo")

    def test_empty(self) -> None:
        assert m.expand_range("") == ()

    @pytest.mark.parametrize("val", ("0", "1", "12", "100"))
    def test_single_number(self, val: str) -> None:
        assert m.expand_range(val) == (int(val),)

    @pytest.mark.parametrize("val", ("0-10", "1-2", "12-25"))
    def test_range(self, val: str) -> None:
        start, stop = val.split("-")
        assert m.expand_range(val) == tuple(range(int(start), int(stop) + 1))


class Test_extract_values:
    def test_errors(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            m.extract_values("foo")

    def test_empty(self) -> None:
        assert m.extract_values("") == ()

    @pytest.mark.parametrize(
        ("val", "expected"),
        [("0", (0,)), ("1,2", (1, 2)), ("3,5,7", (3, 5, 7))],
    )
    def test_individual(self, val: str, expected: tuple[int, ...]) -> None:
        assert m.extract_values(val) == expected

    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            ("2,1", (1, 2)),
            ("8,5,3,2", (2, 3, 5, 8)),
            ("1,3,2,5,4,7,6", (1, 2, 3, 4, 5, 6, 7)),
        ],
    )
    def test_individual_ordered(
        self, val: str, expected: tuple[int, ...]
    ) -> None:
        assert m.extract_values(val) == expected

    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            ("0-2", (0, 1, 2)),
            ("0-2,4-5", (0, 1, 2, 4, 5)),
            ("0-1,3-5,8-11", (0, 1, 3, 4, 5, 8, 9, 10, 11)),
        ],
    )
    def test_range(self, val: str, expected: tuple[int, ...]) -> None:
        assert m.extract_values(val) == expected

    @pytest.mark.parametrize(
        ("val", "expected"),
        [("2-3,0-1", (0, 1, 2, 3)), ("0-1,4-5,2-3", (0, 1, 2, 3, 4, 5))],
    )
    def test_range_ordered(self, val: str, expected: tuple[int, ...]) -> None:
        assert m.extract_values(val) == expected

    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            ("0,1-2", (0, 1, 2)),
            ("1-2,0", (0, 1, 2)),
            ("0,1-2,3,4-5,6", (0, 1, 2, 3, 4, 5, 6)),
            ("5-6,4,1-3,0", (0, 1, 2, 3, 4, 5, 6)),
        ],
    )
    def test_mixed(self, val: str, expected: tuple[int, ...]) -> None:
        assert m.extract_values(val) == expected


class Test_parse_cuda_visible_devices:
    @pytest.mark.parametrize(
        ("env_string", "max_gpu", "expected"), [("0", 8, [0]), ("7", 8, [7])]
    )
    def test_singleton(
        self, env_string: str, max_gpu: int, expected: list[int]
    ) -> None:
        assert m.parse_cuda_visible_devices(env_string, max_gpu) == expected

    @pytest.mark.parametrize(
        ("env_string", "max_gpu", "expected"),
        [("1,2,3", 8, [1, 2, 3]), ("3,1,2", 8, [3, 1, 2])],
    )
    def test_list(
        self, env_string: str, max_gpu: int, expected: list[int]
    ) -> None:
        assert m.parse_cuda_visible_devices(env_string, max_gpu) == expected

    @pytest.mark.parametrize(
        ("env_string", "max_gpu", "expected"),
        [
            ("1,2,-3", 8, [1, 2]),
            ("1,2,x", 8, [1, 2]),
            ("1,2,8", 8, [1, 2]),
            ("1,2,-3,4", 8, [1, 2]),
            ("1,2,x,4", 8, [1, 2]),
            ("1,2,8,4", 8, [1, 2]),
            ("-3,1,2", 8, []),
            ("x,3,4", 8, []),
            ("8,3,4", 8, []),
        ],
    )
    def test_partial(
        self, env_string: str, max_gpu: int, expected: list[int]
    ) -> None:
        assert m.parse_cuda_visible_devices(env_string, max_gpu) == expected


@pytest.mark.skipif(platform.system() == "Darwin", reason="non-OSX test")
def test_linux_cpus_repects_affinity(mocker: MockerFixture) -> None:
    mocker.patch(
        "legate.util.system.linux_load_sibling_sets",
        return_value={(0, 2), (1, 3)},
    )
    mocker.patch("os.sched_getaffinity", return_value={0})

    assert m.System().cpus == (CPUInfo((0, 2)),)
