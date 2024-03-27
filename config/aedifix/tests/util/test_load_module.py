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

import sys
from pathlib import Path
from types import ModuleType

import pytest

from ...util.load_module import load_module_from_path
from ..fixtures import dummy_module


@pytest.fixture
def module_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent / "fixtures" / "dummy_module.py"
    )


class TestLoadModule:
    def do_module_checks(self, mod: ModuleType) -> None:
        # Unfortunately, due to some idiosynchrasies with the loading
        # mechanism, the modules are not the same, because the path-loaded
        # module does not retain any __package__ information. So we must resort
        # to checking magic numbers and attributes to ensure we've loaded the
        # intended module correctly.
        assert mod is not dummy_module
        assert mod.__file__ == dummy_module.__file__
        assert mod.MODULE_ATTRIBUTE == dummy_module.MODULE_ATTRIBUTE
        assert (
            mod.function.__code__.co_filename
            == dummy_module.function.__code__.co_filename
        )
        assert (
            mod.function.MAGIC_NUMBER
            == dummy_module.function.MAGIC_NUMBER  # type: ignore[attr-defined]
        )
        assert mod.Class.MAGIC_ATTR == dummy_module.Class.MAGIC_ATTR

    def test_load_module_from_path(self, module_path: Path) -> None:
        mod = load_module_from_path(module_path)
        self.do_module_checks(mod)

    def test_load_module_from_str(self, module_path: Path) -> None:
        mod = load_module_from_path(str(module_path))
        self.do_module_checks(mod)

    def test_load_module_from_path_bad(self) -> None:
        path = Path("/foo/bar/baz")
        assert not path.exists(), "Well well well..."
        modules_cpy = {k: v for k, v in sys.modules.items()}
        with pytest.raises(
            (ImportError, FileNotFoundError),
            match=rf"\[Errno \d+\] No such file or directory: '{path}'",
        ):
            load_module_from_path(path)

        assert sys.modules == modules_cpy


if __name__ == "__main__":
    sys.exit(pytest.main())
