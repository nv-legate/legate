# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from ..settings import EnvOnlySetting, Settings, convert_bool, convert_str


class BenchmarkSettings(Settings):
    out: EnvOnlySetting[str] = EnvOnlySetting(
        "out",
        "LEGATE_BENCHMARK_OUT",
        default="stdout",
        test_default="stdout",
        convert=convert_str,
        help="""
            Where benchmark_log() records should go if no file is given.

            If this is 'stdout', logs will go to sys.stdout; otherwise this
            will be interpreted as a directory where log files will be created.
            Each rank will save its records in a file with the name
            '{benchmark_name}_{uid}.{node_id}.csv'.

            This is a read-only environment variable setting used by the
            runtime.
            """,
    )
    use_rich: EnvOnlySetting[bool] = EnvOnlySetting(
        "use_rich",
        "LEGATE_BENCHMARK_USE_RICH",
        default=True,
        test_default=False,
        convert=convert_bool,
        help="""
            Whether benchmark_log() should use rich when the output stream
            is a tty

            This is a read-only environment variable setting used by the
            runtime.
            """,
    )


settings = BenchmarkSettings()
