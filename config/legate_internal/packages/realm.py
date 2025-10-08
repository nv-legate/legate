# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from aedifix import Package
from aedifix.cmake import CMAKE_VARIABLE, CMakeBool
from aedifix.util.argument_parser import ArgSpec, ConfigArgument
from aedifix.util.exception import UnsatisfiableConfigurationError

from .gasnet import GASNet
from .mpi import MPI
from .ucx import UCX

if TYPE_CHECKING:
    from aedifix.manager import ConfigurationManager


class Realm(Package):
    name = "Realm"

    dependencies = (GASNet, UCX, MPI)

    REALM_ENABLE_GASNETEX: Final = CMAKE_VARIABLE(
        "REALM_ENABLE_GASNETEX", CMakeBool
    )
    REALM_ENABLE_UCX: Final = CMAKE_VARIABLE("REALM_ENABLE_UCX", CMakeBool)
    REALM_ENABLE_MPI: Final = CMAKE_VARIABLE("REALM_ENABLE_MPI", CMakeBool)
    REALM_ENABLE_GASNETEX_WRAPPER: Final = CMAKE_VARIABLE(
        "REALM_ENABLE_GASNETEX_WRAPPER", CMakeBool
    )
    With_GASNET_WRAPPER: Final = ConfigArgument(
        name="--with-gasnet-wrapper",
        spec=ArgSpec(
            dest="with_gasnet_wrapper",
            type=bool,
            help="Enable Realm GASNetEx ABI wrapper.",
        ),
        cmake_var=REALM_ENABLE_GASNETEX_WRAPPER,
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        super().__init__(manager=manager, always_enabled=True)

    def configure(self) -> None:
        r"""Configure Realm networking options."""
        super().configure()

        self.set_flag_if_user_set(
            self.REALM_ENABLE_GASNETEX_WRAPPER,
            self.cl_args.with_gasnet_wrapper,
        )

        wrapper_enabled = self._cmake_truth(
            self.manager.get_cmake_variable(self.REALM_ENABLE_GASNETEX_WRAPPER)
        )
        if not wrapper_enabled:
            return

        gasnet = cast(GASNet, self.deps.GASNet)
        if not gasnet.state.enabled():
            msg = "--with-gasnet-wrapper requires GASNet to be enabled"
            raise UnsatisfiableConfigurationError(msg)

        if gasnet.cl_args.gasnet_conduit.cl_set:
            conduit = gasnet.cl_args.gasnet_conduit.value
            if conduit != "smp":
                msg = "--gasnet-wrapper requires --gasnet-conduit=smp"
                raise UnsatisfiableConfigurationError(msg)
        else:
            self.manager.set_cmake_variable(gasnet.GASNet_CONDUIT, "smp")

    def summarize(self) -> str:
        r"""Summarize Realm networking."""
        lines = [("Networks", self._network_summary())]

        if self._uses_gasnet():
            gasnet = cast(GASNet, self.deps.GASNet)
            if conduit := self.manager.get_cmake_variable(
                gasnet.GASNet_CONDUIT
            ):
                lines.append(("GASNet conduit", conduit))
            wrapper = self.manager.get_cmake_variable(
                self.REALM_ENABLE_GASNETEX_WRAPPER
            )
            if wrapper is not None:
                lines.append(("GASNetEx wrapper", wrapper))

        return self.create_package_summary(lines)

    def _uses_gasnet(self) -> bool:
        value = self.manager.get_cmake_variable(self.REALM_ENABLE_GASNETEX)
        if value is not None:
            return self._cmake_truth(value)
        return self.deps.GASNet.state.enabled()

    def _uses_ucx(self) -> bool:
        value = self.manager.get_cmake_variable(self.REALM_ENABLE_UCX)
        if value is not None:
            return self._cmake_truth(value)
        return self.deps.UCX.state.enabled()

    def _uses_mpi(self) -> bool:
        value = self.manager.get_cmake_variable(self.REALM_ENABLE_MPI)
        if value is not None:
            return self._cmake_truth(value)
        return self.deps.MPI.state.enabled()

    def _network_summary(self) -> str:
        if self._uses_gasnet():
            return "gasnetex"
        if self._uses_ucx():
            return "ucx"
        if self._uses_mpi():
            return "mpi"
        return "None"

    @staticmethod
    def _cmake_truth(value: object | None) -> bool:
        if isinstance(value, str):
            return value.upper() in {"ON", "1", "TRUE", "YES"}
        return bool(value)
