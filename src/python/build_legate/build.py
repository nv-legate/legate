# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def build_wheel(  # noqa: D103
    wheel_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
    metadata_directory: str | None = None,
) -> str:
    # Imports are done locally as the global namespace of this file is
    # important. Pip will try to import all symbols from this package so we
    # want to export build_wheel and build_wheel *only*.
    from scikit_build_core.build import (  # type: ignore[import-not-found]  # noqa: PLC0415
        build_wheel as orig_build_wheel,
    )

    from .utils import BuildLog, build_impl  # noqa: PLC0415

    with BuildLog(build_kind="wheel"):
        return build_impl(
            orig_impl=orig_build_wheel,
            build_kind="wheel",
            wheel_directory=wheel_directory,
            config_settings=config_settings,
            metadata_directory=metadata_directory,
        )
