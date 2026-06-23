# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from legate.util import info

if TYPE_CHECKING:
    from collections.abc import Iterator


class FakeDist:
    def __init__(
        self,
        name: str = "legate",
        version: str = "26.06.00",
        *,
        files: list[Path] | None = None,
        root: str = "/env/site-packages",
        installer: str | None = "pip\n",
        direct_url: str | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.files = files
        self._root = Path(root)
        self._installer = installer
        self._direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        if name == "INSTALLER":
            return self._installer
        if name == "direct_url.json":
            return self._direct_url
        return None

    def locate_file(self, path: Path) -> Path:
        return self._root / path


@pytest.fixture(autouse=True)
def clear_info_caches() -> Iterator[None]:
    info._package_dists.cache_clear()
    info._conda_package_dists.cache_clear()
    yield
    info._package_dists.cache_clear()
    info._conda_package_dists.cache_clear()


def test_legion_version_is_recorded() -> None:
    legion_version = info.package_versions()["legion"]

    assert legion_version not in ("", info.FAILED_TO_DETECT)
    assert not legion_version.startswith("(")


def test_package_dists_uses_active_python_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find_spec(module_name: str) -> SimpleNamespace | None:
        paths = {
            "legate": "/env/site-packages/legate",
            "cupynumeric": "/env/site-packages/cupynumeric",
        }
        if module_name not in paths:
            return None
        return SimpleNamespace(
            origin=None, submodule_search_locations=[paths[module_name]]
        )

    distributions = {
        "legate": FakeDist(files=[Path("legate/__init__.py")]),
        "nvidia-cupynumeric-cu12": FakeDist(
            "nvidia-cupynumeric-cu12", files=[Path("cupynumeric/__init__.py")]
        ),
    }
    monkeypatch.setattr(info, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        info.importlib_metadata,
        "packages_distributions",
        lambda: {
            "legate": ["legate"],
            "cupynumeric": ["nvidia-cupynumeric-cu12"],
        },
    )
    monkeypatch.setattr(
        info.importlib_metadata,
        "distribution",
        lambda name: distributions[name],
    )

    details = info.package_dists()

    assert details["legate"] == (
        "legate 26.06.00 (installer: pip, path: /env/site-packages/legate)"
    )
    assert details["cupynumeric"] == (
        "nvidia-cupynumeric-cu12 26.06.00 "
        "(installer: pip, path: /env/site-packages/cupynumeric)"
    )


def test_package_dists_requires_package_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        info,
        "find_spec",
        lambda _: SimpleNamespace(
            origin="/src/legate/__init__.py", submodule_search_locations=None
        ),
    )
    monkeypatch.setattr(
        info.importlib_metadata, "packages_distributions", dict
    )

    assert info.package_dists()["legate"] == info.FAILED_TO_DETECT


def test_package_dists_ignores_distribution_that_does_not_own_module_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        info,
        "find_spec",
        lambda _: SimpleNamespace(
            origin=None, submodule_search_locations=["/src/legate"]
        ),
    )
    monkeypatch.setattr(
        info.importlib_metadata,
        "packages_distributions",
        lambda: {"legate": ["legate"]},
    )
    monkeypatch.setattr(
        info.importlib_metadata,
        "distribution",
        lambda _: FakeDist(files=[Path("legate/__init__.py")]),
    )

    assert info.package_dists()["legate"] == (
        f"{info.FAILED_TO_DETECT} (path: /src/legate)"
    )


def test_package_dists_reports_editable_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_url = {"url": "file:///work/legate", "dir_info": {"editable": True}}
    monkeypatch.setattr(
        info,
        "find_spec",
        lambda _: SimpleNamespace(
            origin=None,
            submodule_search_locations=["/work/legate/src/python/legate"],
        ),
    )
    monkeypatch.setattr(
        info.importlib_metadata,
        "packages_distributions",
        lambda: {"legate": ["legate"]},
    )
    monkeypatch.setattr(
        info.importlib_metadata,
        "distribution",
        lambda _: FakeDist(files=None, direct_url=json.dumps(direct_url)),
    )

    assert info.package_dists()["legate"] == (
        "legate 26.06.00 (editable, path: /work/legate/src/python/legate)"
    )


def test_conda_package_dists_reads_active_python_prefix(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path_factory.mktemp("env")
    meta_dir = prefix / "conda-meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "legate-26.06.00-py311_0.json").write_text(
        json.dumps(
            {
                "name": "legate",
                "version": "26.06.00",
                "build": "py311_0",
                "channel": "https://conda.anaconda.org/legate-nightly/linux-64",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(info.sys, "prefix", str(prefix))

    details = info.conda_package_dists()

    assert details["prefix"] == str(prefix)
    assert details["legate"] == (
        "legate-26.06.00-py311_0 "
        "(https://conda.anaconda.org/legate-nightly/linux-64)"
    )
    assert details["cuda-version"] == info.FAILED_TO_DETECT
    assert details["cupynumeric"] == info.FAILED_TO_DETECT


def test_conda_package_dists_reports_missing_active_metadata(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = tmp_path_factory.mktemp("env")
    monkeypatch.setattr(info.sys, "prefix", str(tmp_path))

    details = info.conda_package_dists()

    assert details["prefix"] == str(tmp_path)
    assert details["cuda-version"] == info.NO_CONDA_METADATA
    assert details["legate"] == info.NO_CONDA_METADATA
    assert details["cupynumeric"] == info.NO_CONDA_METADATA
    assert "status" not in details


def test_print_conda_package_dists_collapses_missing_active_metadata(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path = tmp_path_factory.mktemp("env")
    monkeypatch.setattr(info.sys, "prefix", str(tmp_path))

    info.print_conda_package_details()

    out = capsys.readouterr().out
    assert "Conda package details:" in out
    assert "prefix" in out
    assert f"status :  {info.NO_CONDA_METADATA}" in out
    assert "legate    :" not in out
