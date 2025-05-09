# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.optional cimport (
    nullopt as std_nullopt,
    make_optional as std_make_optional
)

from typing import Any

from ..._ext.cython_libcpp.string_view cimport str_from_string_view
from ..mapping.machine cimport Machine
from .exception_mode cimport ExceptionMode
from .runtime cimport raise_pending_exception


cdef class Scope:
    def __init__(
        self,
        *,
        priority: object = None,
        exception_mode: object = None,
        str provenance = None,
        Machine machine = None,
    ) -> None:
        self._priority = (
            _Scope.priority() if priority is None else priority
        )
        self._exception_mode = (
            _Scope.exception_mode()
            if exception_mode is None else <ExceptionMode>exception_mode
        )
        self._provenance = provenance
        self._machine = machine
        self._handle = std_nullopt

    def __enter__(self) -> None:
        if self._handle.has_value():
            raise ValueError(
                "Each Scope object can be used only once as a context manager"
            )

        self._handle = std_make_optional[_Scope]()
        self._handle.value().set_priority(self._priority)
        self._handle.value().set_exception_mode(self._exception_mode)
        if self._provenance is not None:
            self._handle.value().set_provenance(self._provenance.encode())
        if self._machine is not None:
            self._handle.value().set_machine(self._machine._handle)

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        cdef bool exn_deferred = self._exception_mode == ExceptionMode.DEFERRED
        self._handle.reset()
        # If we're about to transition from the deferred exception handling to
        # the eager exception handling, we should raise pending exceptions as
        # the scope collapses.
        if exn_deferred and self.exception_mode() == ExceptionMode.IMMEDIATE:
            raise_pending_exception()

    @staticmethod
    def priority() -> int:
        return _Scope.priority()

    @staticmethod
    def exception_mode() -> ExceptionMode:
        return _Scope.exception_mode()

    @staticmethod
    def provenance() -> str:
        return str_from_string_view(_Scope.provenance())

    @staticmethod
    def machine() -> Machine:
        return Machine.from_handle(_Scope.machine())
