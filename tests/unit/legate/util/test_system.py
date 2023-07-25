# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
