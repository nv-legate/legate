# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

import os
import platform

import pytest
from pytest_mock import MockerFixture

import legate.util.system as m
from legate.util.types import CPUInfo


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
            s.os

    def test_LIBPATH_Linux(self, mocker: MockerFixture) -> None:
        mocker.patch("platform.system", return_value="Linux")

        s = m.System()

        assert s.LIB_PATH == "LD_LIBRARY_PATH"

    def test_LIBPATH_Darwin(self, mocker: MockerFixture) -> None:
        mocker.patch("platform.system", return_value="Darwin")

        s = m.System()

        assert s.LIB_PATH == "DYLD_LIBRARY_PATH"

    # These properties delegate to util functions, just verify plumbing

    def test_legate_paths(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "legate.util.system.get_legate_paths",
            return_value="legate paths",
        )

        s = m.System()

        assert s.legate_paths == "legate paths"  # type: ignore

    def test_legion_paths(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "legate.util.system.get_legion_paths",
            return_value="legion paths",
        )

        s = m.System()

        assert s.legion_paths == "legion paths"  # type: ignore

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
            s.gpus


class Test_expand_range:
    def test_errors(self) -> None:
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            m.extract_values("foo")

    def test_empty(self) -> None:
        assert m.extract_values("") == ()

    testdata_individual = [
        ("0", (0,)),
        ("1,2", (1, 2)),
        ("3,5,7", (3, 5, 7)),
    ]

    @pytest.mark.parametrize("val,expected", testdata_individual)
    def test_individual(self, val: str, expected: tuple[int, ...]) -> None:
        assert m.extract_values(val) == expected

    testdata_individual_ordered = [
        ("2,1", (1, 2)),
        ("8,5,3,2", (2, 3, 5, 8)),
        ("1,3,2,5,4,7,6", (1, 2, 3, 4, 5, 6, 7)),
    ]

    @pytest.mark.parametrize("val,expected", testdata_individual_ordered)
    def test_individual_ordered(
        self, val: str, expected: tuple[int, ...]
    ) -> None:
        assert m.extract_values(val) == expected

    testdata_range = [
        ("0-2", (0, 1, 2)),
        ("0-2,4-5", (0, 1, 2, 4, 5)),
        ("0-1,3-5,8-11", (0, 1, 3, 4, 5, 8, 9, 10, 11)),
    ]

    @pytest.mark.parametrize("val,expected", testdata_range)
    def test_range(self, val: str, expected: tuple[int, ...]) -> None:
        assert m.extract_values(val) == expected

    testdata_range_ordered = [
        ("2-3,0-1", (0, 1, 2, 3)),
        ("0-1,4-5,2-3", (0, 1, 2, 3, 4, 5)),
    ]

    @pytest.mark.parametrize("val,expected", testdata_range_ordered)
    def test_range_ordered(self, val: str, expected: tuple[int, ...]) -> None:
        assert m.extract_values(val) == expected

    testdata_mixed = [
        ("0,1-2", (0, 1, 2)),
        ("1-2,0", (0, 1, 2)),
        ("0,1-2,3,4-5,6", (0, 1, 2, 3, 4, 5, 6)),
        ("5-6,4,1-3,0", (0, 1, 2, 3, 4, 5, 6)),
    ]

    @pytest.mark.parametrize("val,expected", testdata_mixed)
    def test_mixed(self, val: str, expected: tuple[int, ...]) -> None:
        assert m.extract_values(val) == expected


def test_linux_cpus_repects_affinity(mocker: MockerFixture) -> None:
    mocker.patch(
        "legate.util.system.linux_load_sibling_sets",
        return_value={(0, 2), (1, 3)},
    )
    mocker.patch("os.sched_getaffinity", return_value={0})

    assert m.linux_cpus() == (CPUInfo((0, 2)),)
