# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .util.settings import EnvOnlySetting, Settings, convert_bool

__all__ = ("settings",)


class LegateRuntimeSettings(Settings):
    limit_stdout: EnvOnlySetting[bool] = EnvOnlySetting(
        "limit_stdout",
        "LEGATE_LIMIT_STDOUT",
        default=False,
        test_default=False,
        convert=convert_bool,
        help="""
        Whether to limit stdout output to only the first rank.

        This is a read-only environment variable setting used by the runtime.
        """,
    )


settings = LegateRuntimeSettings()
