# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ..runtime.runtime cimport get_legate_runtime


cdef class Trace:
    def __init__(self, uint32_t trace_id):
        self._trace_id = trace_id

    def __enter__(self) -> None:
        get_legate_runtime().begin_trace(self._trace_id)

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        get_legate_runtime().end_trace(self._trace_id)
