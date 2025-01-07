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


def build_wheel(  # noqa: D103
    wheel_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
    metadata_directory: str | None = None,
) -> str:
    from scikit_build_core.build import (  # type: ignore[import-not-found]
        build_wheel as orig_build_wheel,
    )

    from .utils import BuildLog, build_impl

    with BuildLog(build_kind="wheel"):
        return build_impl(
            orig_impl=orig_build_wheel,
            build_kind="wheel",
            wheel_directory=wheel_directory,
            config_settings=config_settings,
            metadata_directory=metadata_directory,
        )
