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

from typing import TYPE_CHECKING

import scikit_build_core.build.wheel  # type: ignore # noqa: PGH003
from scikit_build_core.build._editable import (  # type: ignore # noqa: PGH003
    editable_redirect,
    libdir_to_installed,
    mapping_to_modules,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from scikit_build_core.build._wheelfile import (  # type: ignore  # noqa: PGH003
        WheelWriter,
    )
    from scikit_build_core.settings.skbuild_model import (  # type: ignore # noqa: PGH003
        ScikitBuildSettings,
    )


def patched_make_editable(  # type: ignore # noqa: PGH003, PLR0913
    *,
    build_options: Sequence[str] = (),
    install_options: Sequence[str] = (),
    libdir: Path,
    mapping: dict[str, str],
    name: str,
    reload_dir: Path | None,
    settings: ScikitBuildSettings,
    wheel: WheelWriter,
    packages: Iterable[str],
) -> None:
    modules = mapping_to_modules(mapping, libdir)
    installed = libdir_to_installed(libdir)
    # Our patch
    if False:
        msg = (  # type: ignore[unreachable]
            "Editable installs cannot rebuild an absolute "
            "wheel.install-dir. Use an override to change if needed."
        )
        raise AssertionError(msg)
    editable_txt = editable_redirect(
        modules=modules,
        installed=installed,
        reload_dir=reload_dir,
        rebuild=settings.editable.rebuild,
        verbose=settings.editable.verbose,
        build_options=build_options,
        install_options=install_options,
        install_dir=settings.wheel.install_dir,
    )

    wheel.writestr(f"_{name}_editable.py", editable_txt.encode())
    # Support Cython by adding the source directory directly to the path.
    # This is necessary because Cython does not support sys.meta_path for
    # cimports (as of 3.0.5).
    import_strings = [f"import _{name}_editable", *packages, ""]
    pth_import_paths = "\n".join(import_strings)
    wheel.writestr(f"_{name}_editable.pth", pth_import_paths.encode())


# See https://github.com/scikit-build/scikit-build-core/issues/909
def monkey_patch_skbuild_editable() -> None:
    # Make sure the function still exists
    assert callable(
        scikit_build_core.build.wheel._make_editable  # noqa: SLF001
    )
    scikit_build_core.build.wheel._make_editable = (  # noqa: SLF001
        patched_make_editable
    )
