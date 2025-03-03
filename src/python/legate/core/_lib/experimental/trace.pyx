# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ..runtime.runtime cimport Runtime, get_legate_runtime


cdef class Trace:
    def __init__(self, uint32_t trace_id):
        self._trace_id = trace_id

    def __enter__(self) -> None:
        cdef Runtime runtime = get_legate_runtime()

        runtime._handle.impl().begin_trace(self._trace_id)

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        cdef Runtime runtime = get_legate_runtime()

        runtime._handle.impl().end_trace(self._trace_id)
