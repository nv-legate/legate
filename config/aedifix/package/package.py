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
from __future__ import annotations

import enum
import shlex
from argparse import ArgumentParser, _ArgumentGroup as ArgumentGroup
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from ..base import Configurable
from ..cmake.cmake_flags import _CMakeVar
from ..logger import Logger
from ..util.argument_parser import ConfigArgument, ExclusiveArgumentGroup

if TYPE_CHECKING:
    from ..manager import ConfigurationManager
    from ..util.cl_arg import CLArg


_T = TypeVar("_T")


@dataclass(slots=True, frozen=True)
class EnableState:
    value: bool
    explicit: bool = False

    def enabled(self) -> bool:
        return self.value

    def disabled(self) -> bool:
        return not self.enabled()

    def explicitly_enabled(self) -> bool:
        return self.enabled() and self.explicit

    def explicitly_disabled(self) -> bool:
        return self.disabled() and self.explicit

    def implicitly_enabled(self) -> bool:
        return self.enabled() and (not self.explicit)

    def implicitly_disabled(self) -> bool:
        return self.disabled() and (not self.explicit)

    class Kind(enum.Enum):
        ENABLED = enum.auto()
        DISABLED = enum.auto()
        PRIMARY_DISABLED = enum.auto()


class Package(Configurable):
    __slots__ = "_name", "_state", "_always_enabled"

    def __init__(
        self,
        manager: ConfigurationManager,
        name: str,
        always_enabled: bool = False,
    ) -> None:
        r"""Constuct a Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        name : str
            The name of the package, e.g. 'CUDA'.
        always_enabled : bool, False
            Whether this package should be considered unconditionally enabled.
        """
        super().__init__(manager=manager)
        self._name = name
        self._state = EnableState(value=always_enabled)
        self._always_enabled = always_enabled

    @property
    def name(self) -> str:
        r"""Get the name of the package.

        Returns
        -------
        name : str
            The name of the package, e.g. 'Legion'.
        """
        return self._name

    @property
    def state(self) -> EnableState:
        r"""Get whether the package is enabled or disabled.

        Returns
        -------
        enabled : bool
            True if the package is enabled (i.e. found, requested by user, or
            implied), False otherwise.
        """
        if self._always_enabled:
            assert self._state.enabled(), (
                f"{self.name} is always enabled but state is Falsey: "
                f"{self._state}"
            )
        return self._state

    def force_enable(self) -> None:
        r"""Force a package to be enabled."""
        self.log(f"Forcing {self.name} to always be enabled!")
        self._state = EnableState(value=True, explicit=self.state.explicit)
        self._always_enabled = True

    def add_options(self, parser: ArgumentParser) -> None:
        r"""Add options for a package.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to which to add options.

        Notes
        -----
        Must not be overriden by packages unless they wish to supply no options
        at all.
        """
        group = self.create_argument_group(parser)
        self.log_execute_func(self.add_package_options, group)

    def add_package_options(
        self, parser: ArgumentGroup, ignored_only: bool = False
    ) -> None:
        r"""Callback to add options for each package.

        Parameters
        ----------
        parser : ArgumentGroup
            The argument group to which to add options.

        Notes
        -----
        By default, this does nothing.
        """

        def handle_cmake_var(attr: _CMakeVar) -> None:
            cmake_ty = attr.__config_cmake_type__()
            self.log(
                f'Registering CMake variable "{attr}" for {self!r}: '
                f"{cmake_ty}"
            )
            # have found a special attribute
            self.manager.register_cmake_variable(cmake_ty)

        def handle_config_arg(
            arg: ConfigArgument, parser: ArgumentGroup
        ) -> None:
            self.log(f"Adding {arg.name} to parser: {arg}")
            arg.add_to_argparser(parser)
            if arg.ephemeral:
                self.manager.add_ephemeral_arg(arg.name)
            if arg.cmake_var is not None:
                handle_cmake_var(arg.cmake_var)

        # TODO(jfaibussowit)
        # This is extremely hacky, but basically, for the main package we want
        # to add 2 argument sections: Base Options and <Package Name>
        # Options.
        #
        # So we make 2 argument groups and end up calling this function twice,
        # which results in argparse errors about duplicate argument definitions
        # (because the derived class ends up re-registering the arguments of
        # the parent class).
        #
        # So our solution is as follows, we have a magic attribute
        # __package_ignore_attrs__, which contains the names of all the
        # attributes which should be ignored.
        #
        # So when the main package calls this function, it calls it with
        # ignored_only = True, and only registers its special attributes. The
        # second time around we call it without that, and register all the
        # others.
        attr_names = dir(self)
        ignores: set[str] = getattr(self, "__package_ignore_attrs__", set())
        if ignored_only:
            attr_names = [attr for attr in attr_names if attr in ignores]
        else:
            attr_names = [attr for attr in attr_names if attr not in ignores]

        for attr_name in attr_names:
            if attr_name.startswith("__"):
                continue

            try:
                attr = getattr(self, attr_name)
            except Exception as e:
                self.log(
                    f"Skipping attribute {attr_name!r} due to raised "
                    f"exception: {e}"
                )
                continue

            # attr is e.g. MainPackage.CMAKE_BUILD_TYPE
            match attr:
                case _CMakeVar():
                    self.log(
                        f"Attribute {attr_name!r}: detected cmake variable"
                    )
                    handle_cmake_var(attr)
                case ExclusiveArgumentGroup(required=required, group=group):
                    self.log(
                        f"Attribute {attr_name!r}: detected exclusive "
                        "argument group"
                    )
                    mut_group = parser.add_mutually_exclusive_group(
                        required=required
                    )
                    for sub_attr in group.values():
                        handle_config_arg(sub_attr, mut_group)
                case ConfigArgument():
                    self.log(
                        f"Attribute {attr_name!r}: detected config argument"
                    )
                    handle_config_arg(attr, parser)

    @Logger.log_passthrough
    def require(self, mod_name: str) -> Package:
        r"""Indicate to the manager that `self` requires `mod_name` to run
        before itself, and return a handle to the requested package.

        Parameters
        ----------
        mod_name : str
            The string name of the module. I.e. given `foo.bar.baz.py`, then
            mod_name should be 'baz'

        Returns
        -------
        package : Package
            The indicated package.
        """
        return self.manager.require(self, mod_name)

    def create_argument_group(
        self, parser: ArgumentParser, title: str | None = None
    ) -> ArgumentGroup:
        if title is None:
            title = self.name
        return parser.add_argument_group(title=title)

    def set_flag_if_user_set(
        self, name: str | ConfigArgument, value: CLArg[_T]
    ) -> None:
        if value.cl_set:
            self.manager.set_cmake_variable(name=name, value=value.value)

    def append_flags_if_user_set(
        self, name: str | ConfigArgument, value: CLArg[Sequence[str]]
    ) -> None:
        if value.cl_set:
            self.append_flags_if_set(name, value)

    def append_flags_if_set(
        self, name: str | ConfigArgument, value: CLArg[Sequence[str]]
    ) -> None:
        flags = value.value
        if flags is None:
            return
        assert isinstance(flags, (list, tuple))
        flg_list = []
        for f in flags:
            flg_list.extend(shlex.split(f))
        if not flg_list:
            return
        self.manager.append_cmake_variable(name=name, flags=flg_list)

    # TODO(jfaibussowit)
    # HACK HACK HACK: this is only here because the CUDA package also needs to
    # see it..
    def _configure_language_flags(
        self,
        name: ConfigArgument,
        cl_arg: CLArg[Sequence[str]],
        default_flags: dict[str, list[str]] | None = None,
    ) -> None:
        if default_flags is None:
            from .main_package import _DEFAULT_FLAGS

            default_flags = _DEFAULT_FLAGS[
                self.manager.get_cmake_variable("CMAKE_BUILD_TYPE")
            ]

        if cl_arg.cl_set:
            flags = cl_arg.value
            assert flags is not None
        else:
            flags = default_flags[cl_arg.name]
        self.manager.append_cmake_variable(name=name, flags=flags)

    def find_package(self) -> None:
        r"""Try to locate a package, and enable if it successful. By default,
        does nothing.
        """
        pass

    def _determine_package_enabled(self) -> tuple[EnableState.Kind, bool]:
        r"""Try to determine if a package is enabled or not. By default, simply
        checks the value of --with-<PACKAGE_NAME> and
        --with-<PACKAGE_NAME>-dir.

        Returns
        -------
        enabled : bool
            True if the package is enabled, False otherwise.
        explicit : bool
            True if the state is explicitly set by user, False otherwise
        """
        name = self.name

        if self._always_enabled:
            self.log(f"{name}: always enabled")
            return EnableState.Kind.ENABLED, False

        cl_args = self.cl_args
        lo_name = name.casefold()
        for is_primary_attr, attr_name in (
            (True, f"with_{lo_name}"),
            (False, f"{lo_name}_dir"),
        ):
            self.log(f'{name}: testing for "{attr_name}"')
            if not hasattr(cl_args, attr_name):
                self.log(f'{name}: "{attr_name}", no such attribute exists')
                continue

            attr = getattr(cl_args, attr_name)
            value = attr.value
            if value:
                self.log(
                    f'{name}: enabled due to "{attr_name}" having '
                    f'truthy value "{value}" ({attr})'
                )
                return EnableState.Kind.ENABLED, attr.cl_set

            self.log(
                f'{name}: not enabled due to "{attr_name}" having '
                f'falsey value "{value}" ({attr})'
            )
            if is_primary_attr:
                self.log(
                    f"{name}: stopping attribute search due to primary "
                    f'attribute "{attr_name}" having falsey value'
                )
                return EnableState.Kind.PRIMARY_DISABLED, attr.cl_set

        return EnableState.Kind.DISABLED, False

    def _find_package(self) -> None:
        r"""Attempt to find the current package."""
        enabled_state, user_set = self._determine_package_enabled()
        self._state = EnableState(
            value=enabled_state == EnableState.Kind.ENABLED, explicit=user_set
        )
        if (
            enabled_state != EnableState.Kind.PRIMARY_DISABLED
            and not self.state.explicit
        ):
            # Call subclass hook (whether enabled or disabled), but only if the
            # primary attribute wasn't false and the state was explicitly set.
            self.find_package()
        if self.state.enabled():
            self.log(f"{self.name}: enabled")
        else:
            self.log(
                f"{self.name}: not enabled due to all indicators being falsey"
            )

    def declare_dependencies(self) -> None:
        r"""Set up and declare dependencies for packages. By default,
        declares no dependencies."""
        pass

    def configure(self) -> None:
        r"""Configure a Package."""
        super().configure()
        self.log_execute_func(self._find_package)
        enabled_str = "is" if self.state.enabled() else "is NOT"
        self.log(f"Package {self.name} {enabled_str} enabled")

    def summarize(self) -> str:
        r"""Return a summary of this `Configurable`. By defalt, returns
        an empty summary.

        Returns
        -------
        summary : str
            The summary
        """
        return ""

    def create_package_summary(
        self, extra_lines: Sequence[tuple[str, Any]], title: str | None = None
    ) -> str:
        r"""Create a package summary.

        Parameters
        ----------
        extra_lines : Sequence[tuple[str, Any]]
            Extra lines to add to the package summary.
        title : str, optional
            Title to use for the summary, defaults to package name if not
            given.

        Returns
        -------
        summary : str
            The formatted package summary string.

        Notes
        -----
        Each entry in `extra_lines` must be a pair of values, the line
        heading, and its contents. The line heading must not contain a ';'.
        For example, it may be:

        >>>
            extra_lines = [
                ("Foo", "a foo"),
                ("Bar", "a bar")
            ]

        which results in

        >>>
        Foo: a foo
        Bar: a bar
        """
        if not extra_lines:
            return ""

        if title is None:
            title = self.name

        max_len = max(map(len, (name for name, _ in extra_lines))) + 1
        return "\n".join(
            [f"{title}:"]
            + [
                f"  {str(name) + ':':<{max_len}} {value}"
                for name, value in extra_lines
            ]
        )
