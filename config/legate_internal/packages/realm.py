# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from aedifix.cmake import CMAKE_VARIABLE, CMakeBool, CMakePath
from aedifix.package import Package
from aedifix.util.argument_parser import (
    ArgSpec,
    ConfigArgument,
    ExclusiveArgumentGroup,
)
from aedifix.util.exception import UnsatisfiableConfigurationError
from aedifix.util.utility import dest_to_flag

from .gasnet import GASNet
from .mpi import MPI
from .ucx import UCX

if TYPE_CHECKING:
    from aedifix.manager import ConfigurationManager


class Realm(Package):
    name = "Realm"

    dependencies = (GASNet, UCX, MPI)

    DirGroup: Final = ExclusiveArgumentGroup(
        Realm_ROOT=ConfigArgument(
            name="--with-realm-dir",
            spec=ArgSpec(
                dest="realm_dir",
                type=Path,
                help="Path to an existing Realm build directory.",
            ),
            cmake_var=CMAKE_VARIABLE("Realm_ROOT", CMakePath),
        ),
        CPM_Realm_SOURCE=ConfigArgument(
            name="--with-realm-src-dir",
            spec=ArgSpec(
                dest="with_realm_src_dir",
                type=Path,
                help="Path to an existing Realm source directory.",
            ),
            cmake_var=CMAKE_VARIABLE("CPM_Realm_SOURCE", CMakePath),
        ),
    )
    CPM_DOWNLOAD_Realm: Final = CMAKE_VARIABLE("CPM_DOWNLOAD_Realm", CMakeBool)
    Realm_DIR: Final = CMAKE_VARIABLE("Realm_DIR", CMakePath)
    Realm_BRANCH: Final = ConfigArgument(
        name="--realm-branch",
        spec=ArgSpec(
            dest="realm_branch", help="Git branch to download for Realm"
        ),
    )

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
        enables_package=True,
    )

    def __init__(self, manager: ConfigurationManager) -> None:
        r"""Construct a Realm Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        """
        super().__init__(manager=manager, always_enabled=True)

    def check_conflicting_options(self) -> None:
        r"""Check for conflicting options that are too complicated to "
        "describe statically.

        Raises
        ------
        UnsatisfiableConfigurationError
            If both --with-realm-src-dir and --realm-branch are set
        """
        with_realm_src_dir = self.cl_args.with_realm_src_dir
        realm_branch = self.cl_args.realm_branch
        if with_realm_src_dir.value and realm_branch.value:
            msg = (
                "Cannot specify both "
                f"{dest_to_flag(with_realm_src_dir.name)} and "
                f"{dest_to_flag(realm_branch.name)}, their combined meaning "
                "is ambiguous. If the source dir is given, the contents of "
                "the directory are used as-is (i.e. using whatever branch or "
                "commit that dir is currently on), so the chosen Realm "
                "branch would have no effect."
            )
            raise UnsatisfiableConfigurationError(msg)

    def configure_root_dirs(self) -> None:
        r"""Configure the various "root" directories that Realm requires."""
        dir_group = self.DirGroup
        if (realm_dir := self.cl_args.realm_dir).cl_set:
            self.manager.set_cmake_variable(
                dir_group.Realm_ROOT,  # type: ignore[attr-defined]
                realm_dir.value,
            )
        elif (realm_src_dir := self.cl_args.with_realm_src_dir).cl_set:
            self.manager.set_cmake_variable(
                dir_group.CPM_Realm_SOURCE,  # type: ignore[attr-defined]
                realm_src_dir.value,
            )

    def configure_networking_options(self) -> None:
        r"""Configure Realm networking options."""
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

    def configure(self) -> None:
        r"""Configure Realm."""
        super().configure()
        self.log_execute_func(self.check_conflicting_options)
        self.log_execute_func(self.configure_root_dirs)
        self.log_execute_func(self.configure_networking_options)

    def summarize(self) -> str:
        r"""Summarize Realm.

        Returns
        -------
        summary : str
            A summary of configured Realm.
        """
        m = self.manager

        def get_location() -> Path | None:
            dir_group = self.DirGroup
            root_dir = m.get_cmake_variable(
                dir_group.Realm_ROOT  # type: ignore[attr-defined]
            )
            if root_dir:
                return Path(root_dir)

            root_dir = m.get_cmake_variable(self.Realm_DIR)
            if root_dir:
                return Path(root_dir)

            root_dir = m.get_cmake_variable(
                dir_group.CPM_Realm_SOURCE  # type: ignore[attr-defined]
            )
            if root_dir:
                root_path = Path(root_dir)
                # If the source directory is relative to the cmake
                # directory, then we downloaded Realm, but set
                # CPM_Realm_Source ourselves.
                if not root_path.is_relative_to(m.project_cmake_dir):
                    return root_path
            return None

        lines: list[tuple[str, Any]] = []
        root_dir = get_location()
        downloaded = root_dir is None
        lines.append(("Downloaded", downloaded))
        if not downloaded:
            assert root_dir is not None  # pacify mypy
            lines.append(("  Root dir", root_dir))

        lines.append(("Networks", self._network_summary()))
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
