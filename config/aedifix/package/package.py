# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shlex
import textwrap
from argparse import ArgumentParser, Namespace, _ArgumentGroup as ArgumentGroup
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from ..base import Configurable
from ..cmake.cmake_flags import _CMakeVar
from ..util.argument_parser import ConfigArgument, ExclusiveArgumentGroup
from ..util.exception import WrongOrderError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..manager import ConfigurationManager
    from ..util.cl_arg import CLArg


_T = TypeVar("_T")


class Dependencies(Namespace):
    def __getattr__(self, value: str) -> Package:
        return super().__getattr__(value)


class Package(Configurable):
    __slots__ = "_always_enabled", "_dep_types", "_deps", "_name", "_state"

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

    def __init__(
        self,
        manager: ConfigurationManager,
        name: str,
        *,
        always_enabled: bool = False,
        dependencies: tuple[type[Package], ...] = (),
    ) -> None:
        r"""Construct a Package.

        Parameters
        ----------
        manager : ConfigurationManager
            The configuration manager to manage this package.
        name : str
            The name of the package, e.g. 'CUDA'.
        always_enabled : bool, False
            Whether this package should be considered unconditionally enabled.
        """
        from .main_package import MainPackage

        super().__init__(manager=manager)

        if isinstance(self, MainPackage):
            always_enabled = True

        self._name = name
        self._state = Package.EnableState(value=always_enabled)
        self._always_enabled = always_enabled
        self._dep_types = dependencies
        self._deps: Dependencies | None = None

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

    @property
    def deps(self) -> Dependencies:
        r"""Get the package dependencies.

        Returns
        -------
        deps : Namespace
            The package dependencies.
        """
        if self._deps is None:
            msg = "Must declare dependencies before accessing them"
            raise WrongOrderError(msg)
        return self._deps

    def add_options(self, parser: ArgumentParser) -> None:
        r"""Add options for a package.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to which to add options.

        Notes
        -----
        Must not be overridden by packages unless they wish to supply no
        options at all.
        """
        group = self.create_argument_group(parser)
        self.log_execute_func(self.add_package_options, group)

    def add_package_options(  # noqa: C901
        self, parser: ArgumentGroup, *, ignored_only: bool = False
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
                f'Registering CMake variable "{attr}" for {self!r}: {cmake_ty}'
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

    def create_argument_group(
        self, parser: ArgumentParser, title: str | None = None
    ) -> ArgumentGroup:
        if title is None:
            title = self.name
        return parser.add_argument_group(title=title)

    def set_flag(self, name: str | ConfigArgument, value: CLArg[_T]) -> None:
        self.manager.set_cmake_variable(name=name, value=value.value)

    def set_flag_if_set(
        self, name: str | ConfigArgument, value: CLArg[_T]
    ) -> None:
        if value.value is not None:
            self.set_flag(name=name, value=value)

    def set_flag_if_user_set(
        self, name: str | ConfigArgument, value: CLArg[_T]
    ) -> None:
        if value.cl_set:
            self.set_flag(name=name, value=value)

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

    def declare_dependencies(self) -> None:
        r"""Set up and declare dependencies for packages. By default,
        declares no dependencies.
        """
        deps = {
            dep_ty.__name__: self.manager.require(self, dep_ty)
            for dep_ty in self._dep_types
        }
        self._deps = Dependencies(**deps)

    def _determine_package_enabled(self) -> EnableState:
        r"""Try to determine if a package is enabled or not.

        Returns
        -------
        enabled : EnableState
            Whether the package is enabled.
        """
        if self._always_enabled:
            self.log(f"{self.name}: always enabled")
            return Package.EnableState(value=True, explicit=False)

        config_args = []
        primary_attr = None
        for attr_name in dir(self):
            try:
                attr = getattr(self, attr_name)
            except Exception:  # noqa: S112
                continue

            if not isinstance(attr, ConfigArgument):
                continue

            if not attr.enables_package:
                # Don't care about attributes that don't play a role in
                # enabling the package
                continue

            config_args.append(attr)
            if attr.primary:
                assert primary_attr is None, (
                    "Multiple primary ConfigArgument's, previously "
                    f"found {attr}"
                )
                primary_attr = attr

        assert primary_attr is not None, (
            f"Never found primary config argument for {self.name}"
        )

        # The primary attribute, if set, should ultimately control whether the
        # package is enabled or disabled, so we check it first
        config_args.insert(0, primary_attr)
        for arg in config_args:
            cl_arg = getattr(self.cl_args, arg.spec.dest)
            if (val := cl_arg.value) is not None:
                return Package.EnableState(
                    value=bool(val), explicit=cl_arg.cl_set
                )

        return Package.EnableState(value=False, explicit=False)

    def configure(self) -> None:
        r"""Configure a Package."""
        super().configure()
        self._state = self._determine_package_enabled()
        if self.state.enabled():
            self.log(f"Package {self.name}: enabled")
        else:
            self.log(
                f"Package {self.name}: disabled due to all indicators being "
                "falsey"
            )

    def summarize(self) -> str:
        r"""Return a summary of this `Configurable`. By default, returns
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

        def fixup_extra_lines(
            lines: Sequence[tuple[str, Any]],
        ) -> list[tuple[str, str]]:
            # We want to align any overflow with the start of the text, so
            #
            # foo: some text
            # bar: some ....
            #      very long text
            #      ^^^^^^^^^^^^^^ aligned to "some"
            #
            indent = " " * (max_len + len(":  "))
            ret = []
            for name, value in lines:
                str_v = str(value).strip()
                if "\n" in str_v:
                    str_v = textwrap.indent(str_v, indent).lstrip()
                ret.append((name, str_v))
            return ret

        extra_lines = fixup_extra_lines(extra_lines)
        return "\n".join(
            [f"{title}:"]
            + [
                f"  {str(name) + ':':<{max_len}} {value}"
                for name, value in extra_lines
            ]
        )
