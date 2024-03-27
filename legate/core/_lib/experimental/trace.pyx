# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
