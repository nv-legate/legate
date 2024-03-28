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

import functools
import inspect
import os
import platform
import re
import shutil
import sys
import time
from argparse import (
    SUPPRESS as ARGPARSE_SUPPRESS,
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
)
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from graphlib import TopologicalSorter
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar

from .cmake.cmaker import CMaker
from .config import ConfigFile
from .logger import Logger
from .package import packages
from .reconfigure import Reconfigure
from .util.argument_parser import ConfigArgument
from .util.callables import classify_callable, get_calling_function
from .util.cl_arg import CLArg
from .util.constants import Constants
from .util.exception import WrongOrderError
from .util.types import copy_method_signature
from .util.utility import ValueProvenance, subprocess_capture_output

if TYPE_CHECKING:
    from .cmake.cmake_flags import CMakeFlagBase
    from .package.main_package import MainPackage
    from .package.package import Package

_P = ParamSpec("_P")
_T = TypeVar("_T")


class ConfigurationManager:
    r"""The god-object for a particular configuration. Holds and manages all
    related objects for a run.
    """

    __slots__ = (
        "_cl_args",
        "_argv",
        "_main_package",
        "_modules",
        "_logger",
        "_cmaker",
        "_config",
        "_reconfigure",
        "_ephemeral_args",
        "_module_map",
        "_topo_sorter",
        "_aedifix_root_dir",
    )

    def __init__(
        self, argv: Sequence[str], MainModuleType: type[MainPackage]
    ) -> None:
        r"""Construct a `ConfigurationManager`

        Parameters
        ----------
        argv : Sequence[str]
            The command-line arguments to parse from
        MainModuleType : type[MainPackage]
            The type of the main module for which this manager must produce
            a configuration.
        """
        os.environ["AEDIFIX"] = "1"
        main_package = MainModuleType.from_argv(self, argv)
        self._cl_args: Namespace | None = None
        self._argv = tuple(argv)
        self._main_package = main_package
        self._modules: list[Package] = [main_package]
        self._logger = Logger(self.project_dir / "configure.log")
        self._cmaker = CMaker()
        self._config = ConfigFile(manager=self)
        self._reconfigure = Reconfigure(manager=self)
        self._ephemeral_args: set[str] = set()
        # points to the directory containing the "aedifix" install
        self._aedifix_root_dir: Final = Path(__file__).resolve().parent.parent

    # Private methods
    def _setup_log(self) -> None:
        r"""Output just the ~bear necessities~"""
        self.log_boxed(
            f"Configuring {self.project_name} to compile on your system"
        )
        self.log_divider()
        self.log(
            "Starting configure run at "
            f"{time.strftime('%a, %d %b %Y %H:%M:%S %z')}",
            caller_context=False,
        )
        self.log(
            f"Configure Options: {' '.join(self.argv)}", caller_context=False
        )
        self.log(f"Working directory: {os.getcwd()}", caller_context=False)
        self.log(
            f"Machine platform:\n{platform.uname()}", caller_context=False
        )
        self.log(f"Python version:\n{sys.version}", caller_context=False)
        self.log_divider()
        self.log(
            "Environment Variables:\n"
            + "\n".join(f"{key} = {val}" for key, val in os.environ.items())
        )
        self.log_divider()

    def _parse_args(self, argv: Sequence[str]) -> Namespace:
        r"""Parse arguments as specified in arg list.

        Parameters
        ----------
        argv: Sequence[str]
            The command line arguments to parse.

        Returns
        -------
        args : Namespace
            The parsed arguments.

        Notes
        -----
        Each argument in the returned object lists both the value, and whether
        the argument was explicitly set by the user. This allows detecting
        whether the value is as a result of a default, or whether the user
        specifically set the value.
        """
        parser = ArgumentParser(
            usage="%(prog)s [options...]",
            description=f"Configure {self.project_name}",
            formatter_class=ArgumentDefaultsHelpFormatter,
            # This may lead to confusing errors. E.g.
            #
            # ./configure --cuda --something-else
            #
            # would result in "argument --cuda-arch: expected one argument"
            # because --cuda would disambiguate to --cuda-arch. This would be
            # confusing to the user because clearly they never passed
            # "--cuda-arch" as a flag.
            allow_abbrev=False,
            # We want to catch this as an exception so we can properly log it.
            exit_on_error=False,
        )

        for conf_obj in self._modules:
            self.log_execute_func(conf_obj.add_options, parser)

        # Parse the arguments normally, this will populate the values from the
        # defaults
        if "-h" in argv or "--help" in argv:
            self.log("", tee=True, caller_context=False)  # to undo scrolling

        full_args = parser.parse_args(argv)
        # Create a dummy parser which will determine whether a particular value
        # was passed on the command line. This is done by re-adding all of the
        # arguments in the parsed args, but making the default
        # value "suppressed".
        #
        # If an arguments default is suppressed, then if that argument does NOT
        # appear in the command line, then it is NOT added to the resultant
        # namespace. Thus any argument that exists in both the full args and
        # the suppressed args implies that argument was set by the user.
        suppress_parser = ArgumentParser(
            argument_default=ARGPARSE_SUPPRESS, add_help=False
        )
        for action in parser._actions:
            suppress_parser.add_argument(
                *action.option_strings, dest=action.dest, nargs="*"
            )
        cli_args, _ = suppress_parser.parse_known_args(argv)

        args = Namespace()
        for name, value in vars(full_args).items():
            setattr(
                args,
                name,
                CLArg(name=name, value=value, cl_set=hasattr(cli_args, name)),
            )
        return args

    def _setup_environ(self) -> None:
        r"""Sets up the environment. Among other things, properly injects
        the values of PROJECT_ARCH and PROJECT_DIR if unset.
        """
        arch_name = self.project_arch_name
        arch_value = self.project_arch
        environ = os.environ
        match self._main_package.arch_value_provenance:
            case ValueProvenance.COMMAND_LINE:
                if arch_name in environ:
                    if (env_val := environ[arch_name]) != arch_value:
                        self.log_warning(
                            f"Ignoring environment variable "
                            f'{arch_name}="{env_val}". Using command-line '
                            f'value "{arch_value}" instead.'
                        )
                else:
                    self.log(
                        f'Using {arch_name} from command-line: "{arch_value}"'
                    )
            case ValueProvenance.ENVIRONMENT:
                assert arch_name in environ, (
                    f"Arch provenance was environment, but {arch_name} not "
                    "found in os.environ"
                )
                self.log(
                    f"{arch_name} found in environment: "
                    f'"{environ[arch_name]}"'
                )
            case ValueProvenance.GENERATED:
                self.log(f"{arch_name} was generated")

        self.log(
            f"Setting environment value for {arch_name}, new value: "
            f"{arch_value}"
        )
        environ[arch_name] = arch_value
        dir_name = self.project_dir_name
        dir_value = self.project_dir
        self.log(
            f"Setting environment value for {dir_name}, new value: {dir_value}"
        )
        environ[dir_name] = str(dir_value)

    def _setup_arch_dir(self, clean_first: bool) -> None:
        r"""Ensure the creation and validity of the project arch directory.

        Parameters
        ----------
        clean_first : bool
            `True` if the arch directory should be cleared first, `False`
            otherwise.

        Raises
        ------
        RuntimeError
            If the arch directory exists but is not a directory.
        """
        arch_dir = self.project_arch_dir
        proj_name = self.project_name
        if clean_first:
            self.log_warning(
                f"--with-clean specified, deleting contents of {arch_dir}!"
            )
        if arch_dir.exists():
            self.log(f"{proj_name} arch exists: {arch_dir}")
            if not arch_dir.is_dir():
                raise RuntimeError(
                    f"{proj_name} arch directory "
                    f"{arch_dir} already exists, but is not a "
                    "directory. Please delete move or delete this file "
                    "before re-running configure!"
                )
            if not clean_first:
                self.log(f"Successfully setup arch directory: {arch_dir}")
                return

            self.log("Deleting arch directory, then recreating it")
            proj_dir = self.project_dir
            if arch_dir == proj_dir or arch_dir in proj_dir.parents:
                # yes, this happened to me :(
                raise RuntimeError(
                    f"Arch dir {arch_dir} is either a sub-path of or is the "
                    f"same as the project dir ({proj_dir}). Deleting the arch "
                    "dir would be catastrophic, probably this is a mistake!"
                )

            self.log_execute_func(self._reconfigure.backup_reconfigure_script)
            shutil.rmtree(arch_dir)
        self.log(f"{proj_name} arch does not exist, creating {arch_dir}")
        arch_dir.mkdir(parents=True)
        self.log(f"Successfully setup arch directory: {arch_dir}")

    def _setup_dependencies(self) -> None:
        r"""Setup the package dependency tree.

        Notes
        -----
        This function is the only place where packages may declare
        dependencies. After this function returns, self._modules is sorted in
        topological order based on the requirements dictated by the packages.
        """

        @functools.cache
        def conf_name_cache_get(conf_obj: Package) -> str:
            module = inspect.getmodule(conf_obj)
            assert module is not None
            long_name = module.__name__
            if pack_name := module.__package__:
                short_name = long_name.replace(pack_name, "")
            else:
                short_name = long_name
            return short_name.lstrip(".")

        mod_idx_map: dict[str, int] = {}
        for idx, conf_obj in enumerate(self._modules):
            short_name = conf_name_cache_get(conf_obj)
            assert short_name not in mod_idx_map, (
                # TODO, there can be conflicts if we have foo.bar and baz.bar
                f"Duplicate modules in module map: {short_name} "
                f"({idx} and {mod_idx_map[short_name]}). "
                "Should make this logic more robust!"
            )
            mod_idx_map[short_name] = idx

        self._module_map = mod_idx_map
        assert len(self._module_map) == len(
            self._modules
        ), "Duplicate modules!"

        # pre-populate the topologicalsorter so that modules which are never
        # "required" are properly encoded with no dependencies
        self._topo_sorter = TopologicalSorter(
            {conf_obj: {} for conf_obj in self._modules}
        )

        for conf_obj in self._modules:
            self.log_execute_func(conf_obj.declare_dependencies)

        # need to regen self._modules, but also reorder the module map since
        # modules may have changed order
        self._modules = []
        for idx, conf_obj in enumerate(self._topo_sorter.static_order()):
            self._modules.append(conf_obj)
            self._module_map[conf_name_cache_get(conf_obj)] = idx

        del self._topo_sorter

    def _get_package(self, req_package: str) -> Package:
        try:
            ret_idx = self._module_map[req_package]
        except KeyError as ke:
            raise ModuleNotFoundError(req_package) from ke

        return self._modules[ret_idx]  # should should never fail

    # Member variable access
    @property
    def argv(self) -> tuple[str, ...]:
        r"""Get the parsed command-line arguments.

        Returns
        -------
        args : Namespace
            The parsed command-line arguments.

        Notes
        -----
        Can only be called after `ConfigurationManager.setup()`.
        """
        return self._argv

    @property
    def cl_args(self) -> Namespace:
        r"""Get the parsed command-line arguments.

        Returns
        -------
        args : Namespace
            The parsed command-line arguments.

        Raises
        ------
        WrongOrderError
            If the attribute is retrieved before
            `ConfigurationManager.setup()` is called.
        """
        if self._cl_args is None:
            raise WrongOrderError("Must call setup() first")
        return self._cl_args

    @property
    def project_name(self) -> str:
        r"""Get the name of the current main project.

        Returns
        -------
        name : str
            The name of the current main project, e.g. "Legate.Core".
        """
        return self._main_package.name

    @property
    def project_arch(self) -> str:
        r"""Get the current main project arch.

        Returns
        -------
        arch : str
            The arch name of the current main project,
            e.g. "arch-darwin-debug".
        """
        return self._main_package.arch_value

    @property
    def project_arch_name(self) -> str:
        r"""Get the current main project arch flag name.

        Returns
        -------
        flag_name : str
            The name of the current main project arch flag,
            e.g. "LEGATE_CORE_ARCH".
        """
        return self._main_package.arch_name

    @property
    def project_dir(self) -> Path:
        r"""Get the current main project root directory.

        Returns
        -------
        dir : Path
            The full path to the current project root directory, e.g.
            `/path/to/legate.core.internal`.
        """
        return self._main_package.project_dir_value

    @property
    def project_dir_name(self) -> str:
        r"""Get the name of the current main project root directory.

        Returns
        -------
        dir_name : Path
            The name of the current project root directory,
            e.g. "LEGATE_CORE_DIR".
        """
        return self._main_package.project_dir_name

    @property
    def project_arch_dir(self) -> Path:
        r"""Get the the current main project arch directory.

        Returns
        -------
        arch_dir : Path
            The full path to the current project arch directory,
            e.g. `/path/to/legate.core.internal/arch-darwin-debug`.
        """
        return self.project_dir / self.project_arch

    @property
    def project_cmake_dir(self) -> Path:
        r"""Get the projects current cmake directory.

        Returns
        -------
        cmake_dir : Path
            The full path to the current project cmake directory.
            e.g. `/path/to/legate.core.internal/arch-darwin-debug/cmake_build`.
        """
        return self.project_arch_dir / "cmake_build"

    @staticmethod
    def _sanitize_name(var: str | ConfigArgument) -> str:
        name: str
        if isinstance(var, ConfigArgument):
            if var.cmake_var is None:
                raise ValueError(
                    f"CMake Variable for {var.name} is unset: {var}"
                )
            name = var.cmake_var
        else:
            name = var
        return name

    # CMake variables
    @Logger.log_passthrough
    def register_cmake_variable(self, var: CMakeFlagBase) -> None:
        self._cmaker.register_variable(self, var)

    @Logger.log_passthrough
    def set_cmake_variable(
        self, name: str | ConfigArgument, value: Any
    ) -> None:
        self._cmaker.set_value(self, self._sanitize_name(name), value)

    @Logger.log_passthrough
    def get_cmake_variable(self, name: str | ConfigArgument) -> Any:
        return self._cmaker.get_value(self, self._sanitize_name(name))

    @Logger.log_passthrough
    def append_cmake_variable(
        self, name: str | ConfigArgument, flags: Sequence[str]
    ) -> Any:
        return self._cmaker.append_value(
            self, self._sanitize_name(name), flags
        )

    def read_cmake_variable(self, name: str | ConfigArgument) -> str:
        r"""Read a CMake variable from the cache.

        Parameters
        ----------
        name : str
            The name of the CMake variable to read.

        Returns
        -------
        value : str
            The value of the CMake variable.

        Raises
        ------
        ValueError
            If the value could not be found.
        """
        name = self._sanitize_name(name)
        cmake_cache_txt = self.project_cmake_dir / "CMakeCache.txt"
        re_pat = re.compile(name)
        with cmake_cache_txt.open() as fd:
            for line in filter(re_pat.match, fd):
                return line.split("=")[1].strip()

        raise ValueError(f"Did not find {name} in {cmake_cache_txt}")

    # Rules
    @copy_method_signature(ConfigFile.add_rule)
    def add_gmake_rule(  # type: ignore[no-untyped-def] # copied signature
        self, *args, **kwargs
    ) -> None:
        self._config.add_rule(*args, **kwargs)

    @copy_method_signature(ConfigFile.add_variable)
    def add_gmake_variable(  # type: ignore[no-untyped-def] # copied signature
        self, *args, **kwargs
    ) -> None:
        self._config.add_variable(*args, **kwargs)

    @copy_method_signature(ConfigFile.add_search_variable)
    def add_gmake_search_variable(  # type: ignore[no-untyped-def] # copied sig
        self, *args, **kwargs
    ) -> None:
        self._config.add_search_variable(*args, **kwargs)

    # Logging
    def log(
        self,
        mess: str,
        *,
        tee: bool = False,
        end: str = "\n",
        scroll: bool = False,
        caller_context: bool = True,
    ) -> None:
        r"""Append a message to the log.

        Parameters
        ----------
        mess : str
            The message to append to the log.
        tee : bool, False
            If True, output is printed to screen in addition to being appended
            to the on-disk log file. If False, output is only written to disk.
        end : str, '\n'
            The line ending to append to the message.
        scroll : bool, False
            If True, writes `mess` on a new line, if False overwrites the
            current line with `mess`.
        caller_context : bool, True
            Whether to prepand the name of the function which called this
            function to `mess`.
        """
        verbose_mess = mess
        if caller_context:
            try:
                caller = get_calling_function()
            except ValueError:
                pass
            else:
                caller_name, _, _ = classify_callable(
                    caller, fully_qualify=False
                )
                verbose_mess = f"{caller_name}(): {mess}"

        self._logger.log_file(verbose_mess)
        if not tee:
            return

        if scroll:
            self._logger.log_screen_clear_line()
            mess = mess[: Constants.banner_length]
            end = "\r"
        self._logger.log_screen(mess, end=end)

    @Logger.log_passthrough
    def log_divider(self, tee: bool = False) -> None:
        r"""Append a dividing line to the logs.

        Parameters
        ----------
        tee : bool, False
           If True, output is printed to screen in addition to being appended
           to the on-disk log file. If False, output is only written to disk.
        """
        self.log("=" * Constants.banner_length, tee=tee, caller_context=False)

    @Logger.log_passthrough
    def log_boxed(
        self,
        message: str,
        *,
        title: str = "",
        divider_char: str | None = None,
        tee: bool = True,
        caller_context: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""Log a message surrounded by a box.

        Parameters
        ----------
        message : str
            The message to log.
        title : str, ''
            An optional title for the box.
        divider_char : str | None
            The character to use as the divider between the title and box
            contents.
        tee : bool, True
            If True, output is printed to screen in addition to being appended
            to the on-disk log file. If False, output is only written to disk.
        caller_context : bool, False
            Same meaning as for `ConfigurationManager.log()`.
        **kwargs : Any
            Additional keyword arguments to `ConfigurationManager.log()`.
        """
        self.log_divider(tee=tee)
        if divider_char is None:
            if title:
                divider_char = "-"
        else:
            assert (
                len(divider_char) == 1
            ), "divider CHAR must be a char (i.e. length 1), not a string!"
        message = self._logger.build_multiline_message(
            title, message, divider_char=divider_char
        )
        self.log(message, tee=tee, caller_context=caller_context, **kwargs)
        self.log_divider(tee=tee)

    @Logger.log_passthrough
    def log_warning(
        self, message: str, *, title: str = "WARNING", **kwargs: Any
    ) -> None:
        r"""Log a warning to the log.

        Parameters
        ----------
        message : str
            The message to print.
        title : str, 'WARNING'
            The title to use for the box.
        **kwargs : Any
            Keyword arguments to `ConifgurationManager.log_boxed()`.
        """
        self.log_boxed(message, title=f"***** {title} *****", **kwargs)

    def log_execute_command(
        self, command: Sequence[_T]
    ) -> CompletedProcess[str]:
        r"""Execute a system command and return the output.

        Parameters
        ----------
        command : Sequence[T]
            The command list to execute.

        Returns
        -------
        ret : CompletedProcess
            The completed process object.

        Raises
        ------
        RuntimeError
            If the command returns a non-zero errorcode
        """
        self.log(f"Executing command: {' '.join(map(str, command))}")
        try:
            ret = subprocess_capture_output(command, check=True)
        except RuntimeError as rte:
            self.log(str(rte))
            raise

        self.log(f"STDOUT:\n{ret.stdout}")
        self.log(f"STDERR:\n{ret.stderr}")
        return ret

    def log_execute_func(
        self, fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        r"""Execute a function and log its execution to screen and log.

        Parameters
        ----------
        fn : Callable[P, T]
            The callable object to execute.
        *args : P.args
            The positional arguments to `fn`.
        **kwargs : P.kwargs
            The keyword arguments to `fn`.

        Returns
        -------
        ret : T
            The return value of `fn`.
        """

        def pruned_path(src_path: Path) -> Path:
            # Given
            #
            # src_path = '/path/to/project/foo/bar/baz/module.py
            #
            # we want to extract 'foo/bar/baz/module.py' since that makes for
            # prettier printing below
            for parent in (self._aedifix_root_dir, self.project_dir):
                try:
                    return src_path.relative_to(parent)
                except ValueError:
                    continue
            return src_path

        qual_name, src_path, lineno = classify_callable(fn)
        qual_path = pruned_path(src_path)
        self.log_divider()
        self.log(
            f"RUNNING: {qual_name}() ({qual_path}:{lineno})",
            tee=True,
            scroll=True,
            caller_context=False,
        )
        if docstr := inspect.getdoc(fn):
            self.log(f"  {docstr}\n", caller_context=False)
        else:
            # for a newline
            self.log("", caller_context=False)
        return fn(*args, **kwargs)

    # Meat and potatoes
    def require(self, package: Package, req_package: str) -> Package:
        r"""Indicate to the manager that `package` requires `req_package` to
        run before itself, and return a handle to the requested package.

        Parameters
        ----------
        package : Package
            The package that is requesting the dependency.
        req_package : str
            The string name of the module. I.e. given `foo.bar.baz.py`, then
            mod_name should be 'baz'.

        Returns
        -------
        package : Package
            The indicated package.

        Raises
        ------
        RuntimeError
            If this routine is called outside of
            `ConfigurationManager.setup_dependencies()`.
        ModuleNotFoundError
            If the requested module cannot be located.
        """
        self.log(f"Module {package} requesting requirement: {req_package}")
        ret = self._get_package(req_package)
        try:
            topo_sorter = self._topo_sorter
        except AttributeError as ae:
            raise RuntimeError(
                "Trying to require a module outside of setup_dependencies(), "
                "this is not allowed"
            ) from ae

        topo_sorter.add(package, ret)
        return ret

    def add_ephemeral_arg(self, arg: str) -> None:
        r"""Register an ephemeral command-line argument.

        Parameters
        ----------
        arg : str
            The command-line argument to add.

        Notes
        -----
        Ephemeral arguments are a set of "one-shot" arguments, which that
        should not re-appear on a reconfiguration run.
        """
        self._ephemeral_args.add(arg)

    def setup(self) -> None:
        r"""Setup the `ConfigurationManager`, and parse any command line
        arguments.

        Notes
        -----
        This routine will also ensure the creation of the arch directory.
        """
        self._setup_log()
        self._modules.extend(
            self.log_execute_func(packages.load_packages, self)
        )

        self.log_execute_func(
            self._main_package.inspect_packages, self._modules
        )

        # Sort the modules alphabetically for the parsing of arguments, but
        # keep the main package on top.
        self._modules.remove(self._main_package)
        self._modules.sort(key=lambda x: x.name.casefold())
        self._modules.insert(0, self._main_package)

        self._cl_args = self.log_execute_func(self._parse_args, self.argv)
        # do this after parsing args because args might have --help (in which
        # case we do not want to clobber the arch directory)
        self.log_execute_func(self._setup_environ)
        self.log_execute_func(
            self._setup_arch_dir, self.cl_args.with_clean.value
        )
        # This call re-shuffles the modules
        self.log_execute_func(self._setup_dependencies)
        self.log_execute_func(self._config.setup)
        self.log_execute_func(self._reconfigure.setup)

        for conf in self._modules:
            self.log_execute_func(conf.setup)

    def configure(self) -> None:
        r"""Configure all collected modules."""
        self.log_execute_func(self._config.configure)
        self.log_execute_func(self._reconfigure.configure)
        for conf in self._modules:
            self.log_execute_func(conf.configure)

    def finalize(self) -> None:
        r"""Finalize the configuration and instantiate the CMake configure."""
        for conf_obj in self._modules:
            self.log_execute_func(conf_obj.finalize)

        self.log_execute_func(
            self._cmaker.finalize,
            self,
            self.project_dir,
            self.project_cmake_dir,
        )

        self.log_execute_func(self._config.finalize)
        self.log_execute_func(
            self._reconfigure.finalize,
            main_package_type=type(self._main_package),
            ephemeral_args=self._ephemeral_args,
        )

        def gen_summary() -> Iterator[str]:
            summary = defaultdict(list)
            summary[self._main_package.name].append(
                self.log_execute_func(self._main_package.summarize_main)
            )
            for conf_obj in self._modules:
                if ret := self.log_execute_func(conf_obj.summarize):
                    summary[conf_obj.name].append(ret)

            for val_list in summary.values():
                yield from val_list

        summary = "\n".join(gen_summary())
        self.log_divider(tee=True)
        self.log(summary, tee=True, caller_context=False)
        self.log_boxed(
            "\n".join(
                [
                    "Please set the following:",
                    "",
                    f"export {self.project_arch_name}='{self.project_arch}'",
                    f"export {self.project_dir_name}='{self.project_dir}'",
                    "",
                    "Then build libraries:",
                    "$ make",
                ]
            ),
            title="Configuration Complete",
        )
        self._logger.copy_log(self.project_arch_dir / "configure.log")
