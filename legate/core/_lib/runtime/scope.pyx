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

from libc.stdint cimport int32_t
from libcpp.optional cimport nullopt as std_nullopt

from typing import Any

from ..legate_c cimport _LEGATE_CORE_DEFAULT_TASK_PRIORITY
from ..mapping.machine cimport Machine


cdef class Scope:
    def __init__(
        self,
        *,
        int32_t priority = _LEGATE_CORE_DEFAULT_TASK_PRIORITY,
        str provenance = None,
        Machine machine = None,
    ) -> None:
        self._priority = priority
        self._provenance = provenance
        self._machine = machine
        self._handle = std_nullopt

    def __enter__(self) -> None:
        if self._handle.has_value():
            raise ValueError(
                "Each Scope object can be used only once as a conetxt manager"
            )

        self._handle = _Scope()
        self._handle.value().set_priority(self._priority)
        if self._provenance is not None:
            self._handle.value().set_provenance(self._provenance.encode())
        if self._machine is not None:
            self._handle.value().set_machine(self._machine._handle)

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        self._handle.reset()

    @staticmethod
    def priority() -> int:
        return _Scope.priority()

    @staticmethod
    def provenance() -> str:
        return _Scope.provenance().decode()

    @staticmethod
    def machine() -> Machine:
        return Machine.from_handle(_Scope.machine())
