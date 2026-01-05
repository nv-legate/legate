# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def build_editable(  # noqa: D103
    wheel_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
    metadata_directory: str | None = None,
) -> str:
    # Imports are done locally as the global namespace of this file is
    # important. Pip will try to import all symbols from this package so we
    # want to export build_editable and build_editable *only*.
    import os  # noqa: PLC0415

    from scikit_build_core.build import (  # type: ignore[import-not-found]  # noqa: PLC0415
        build_editable as orig_build_editable,
    )

    from .utils import BuildLog, build_impl, vprint  # noqa: PLC0415

    with BuildLog(build_kind="editable"):
        do_patch = os.environ.get("LEGATE_PATCH_SKBUILD", "1").strip() == "1"
        vprint("patching skbuild_core.build.wheel._make_editable:", do_patch)
        if do_patch:
            from .utils._monkey import (  # noqa: PLC0415
                monkey_patch_skbuild_editable,
            )

            monkey_patch_skbuild_editable()

        return build_impl(
            orig_impl=orig_build_editable,
            build_kind="editable",
            wheel_directory=wheel_directory,
            config_settings=config_settings,
            metadata_directory=metadata_directory,
        )
