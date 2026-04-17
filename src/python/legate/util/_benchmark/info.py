# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import datetime
import platform
from typing import TYPE_CHECKING, TypedDict

from ..has_started import runtime_has_started

if TYPE_CHECKING:
    import numpy as np

BenchmarkInfo = TypedDict(
    "BenchmarkInfo",
    {
        "Name": str,
        "ID": str,
        "Time": datetime.datetime,
        "Global rank": int,
        "Local rank": int,
        "Global size": int | str,
        "Node": str,
    },
)


def benchmark_info(name: str, uid: np.uint64) -> BenchmarkInfo:
    """Create a dictionary of info about a benchmark run.

    Parameters
    ----------
    name: str
        The name of the benchmark.
    uid: np.uint64
        The unique identifier of the benchmark.

    Returns
    -------
    BenchmarkInfo
        This data supplements `legate.util.info.info()` with information
        specific to the benchmark and when/where it is being run.
    """
    global_size: int | str = "unknown"
    if runtime_has_started():
        # local import to avoid starting the runtime unless requested
        from legate.core import get_legate_runtime  # noqa: PLC0415

        runtime = get_legate_runtime()
        machine = runtime.get_machine()
        nodes = machine.get_node_range()
        global_size = nodes[1] - nodes[0]
    return {
        "Name": name,
        "ID": f"{uid:016x}",
        "Time": datetime.datetime.now(datetime.UTC).replace(microsecond=0),
        "Global rank": int(os.getenv("LEGATE_GLOBAL_RANK", default="0")),
        "Local rank": int(os.getenv("LEGATE_LOCAL_RANK", default="0")),
        "Global size": global_size,
        "Node": platform.node(),
    }
