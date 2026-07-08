# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.optional cimport (
    nullopt as std_nullopt,
    make_optional as std_make_optional
)

from typing import Any

from ..._ext.cython_libcpp.string_view cimport str_from_string_view
from ..mapping.machine cimport Machine
from .parallel_policy cimport ParallelPolicy


cdef class Scope:
    def __init__(
        self,
        *,
        priority: object = None,
        str provenance = None,
        Machine machine = None,
        ParallelPolicy parallel_policy = None,
    ) -> None:
        self._priority = (
            _Scope.priority() if priority is None else priority
        )
        self._provenance = provenance
        self._machine = machine
        self._parallel_policy = parallel_policy
        self._handle = std_nullopt

    def __enter__(self) -> None:
        if self._handle.has_value():
            raise ValueError(
                "Each Scope object can be used only once as a context manager"
            )

        self._handle = std_make_optional[_Scope]()
        self._handle.value().set_priority(self._priority)
        if self._provenance is not None:
            self._handle.value().set_provenance(self._provenance.encode())
        if self._machine is not None:
            self._handle.value().set_machine(self._machine._handle)
        if self._parallel_policy is not None:
            self._handle.value().set_parallel_policy(
                self._parallel_policy._handle
            )

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        self._handle.reset()

    @staticmethod
    def priority() -> int:
        return _Scope.priority()

    @staticmethod
    def provenance() -> str:
        return str_from_string_view(_Scope.provenance())

    @staticmethod
    def machine() -> Machine:
        return Machine.from_handle(_Scope.machine())

    @staticmethod
    def parallel_policy() -> ParallelPolicy:
        r"""
        This allows users to get the parallel_policy of this Scope

        Returns
        -------
        ParallelPolicy
            The parallel_policy of this Scope.
        """
        return ParallelPolicy.from_handle(_Scope.parallel_policy())
