# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def build_editable(  # noqa: D103
    wheel_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
    metadata_directory: str | None = None,
) -> str:
    import os

    from scikit_build_core.build import (  # type: ignore[import-not-found]
        build_editable as orig_build_editable,
    )

    from .utils import BuildLog, build_impl, vprint

    with BuildLog(build_kind="editable"):
        do_patch = os.environ.get("LEGATE_PATCH_SKBUILD", "1").strip() == "1"
        vprint("patching skbuild_core.build.wheel._make_editable:", do_patch)
        if do_patch:
            from .utils._monkey import monkey_patch_skbuild_editable

            monkey_patch_skbuild_editable()

        return build_impl(
            orig_impl=orig_build_editable,
            build_kind="editable",
            wheel_directory=wheel_directory,
            config_settings=config_settings,
            metadata_directory=metadata_directory,
        )
