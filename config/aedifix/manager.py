# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
import time
import shutil
import inspect
import platform
import textwrap
from argparse import (
    SUPPRESS as ARGPARSE_SUPPRESS,
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    RawDescriptionHelpFormatter,
)
from collections import defaultdict
from graphlib import TopologicalSorter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar

from .cmake.cmaker import CMaker
from .config import ConfigFile
from .logger import Logger
from .package import packages
from .reconfigure import Reconfigure
from .util.argument_parser import ConfigArgument
from .util.callables import classify_callable, get_calling_function
from .util.cl_arg import CLArg
from .util.exception import (
    CommandError,
    UnsatisfiableConfigurationError,
    WrongOrderError,
)
from .util.utility import (
    ValueProvenance,
    dest_to_flag,
    partition_argv,
    subprocess_capture_output_live,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from subprocess import CompletedProcess

    from .cmake.cmake_flags import CMakeFlagBase
    from .logger import AlignMethod
    from .package.main_package import MainPackage
    from .package.package import Package

_P = ParamSpec("_P")
_T = TypeVar("_T")


class ConfigurationManager:
    r"""The god-object for a particular configuration. Holds and manages all
    related objects for a run.
    """

    __slots__ = (
        "_aedifix_root_dir",
        "_argv",
        "_cl_args",
        "_cmaker",
        "_config",
        "_ephemeral_args",
        "_extra_argv",
        "_logger",
        "_main_package",
        "_module_map",
        "_modules",
        "_orig_argv",
        "_reconfigure",
        "_topo_sorter",
    )

    def __init__(
        self, argv: Sequence[str], MainModuleType: type[MainPackage]
    ) -> None:
        r"""Construct a `ConfigurationManager`.

        Parameters
        ----------
        argv : Sequence[str]
            The command-line arguments to parse from
        MainModuleType : type[MainPackage]
            The type of the main module for which this manager must produce
            a configuration.
        """
        os.environ["AEDIFIX"] = "1"
        # points to the directory containing the "aedifix" install
        self._aedifix_root_dir: Final = Path(__file__).resolve().parent.parent
        main_package = MainModuleType.from_argv(self, argv)
        self._cl_args: Namespace | None = None
        self._orig_argv = tuple(argv)
        main_argv, extra_argv = partition_argv(self._orig_argv)
        self._argv = tuple(main_argv)
        self._extra_argv = extra_argv
        self._main_package = main_package
        self._modules: list[Package] = [main_package]
        self._logger = Logger(self.project_dir / "configure.log")
        self._cmaker = CMaker()
        self._config = ConfigFile(
            manager=self,
            config_file_template=main_package.project_configure_file_template,
        )
        self._reconfigure = Reconfigure(manager=self)
        self._ephemeral_args: set[str] = set()

    # Private methods
    def _setup_log(self) -> None:
        r"""Output just the ~bear necessities~."""
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
        self.log(f"Working directory: {Path.cwd()}", caller_context=False)
        self.log(
            f"Machine platform:\n{platform.uname()}", caller_context=False
        )
        self.log(f"Python version:\n{sys.version}", caller_context=False)
        self.log_divider()
        self.log(
            "Environment Variables:\n"
            + "\n".join(
                f"{key} = {val}" for key, val in sorted(os.environ.items())
            )
        )
        self.log_divider()

    def _log_git_info(self) -> None:
        r"""Log information about the current commit and branch of the
        repository.
        """
        git_exe = shutil.which("git")
        if git_exe is None:
            self.log(
                "'git' command not found, likely not a development repository"
            )
            return

        try:
            branch = self.log_execute_command(
                [git_exe, "branch", "--show-current"]
            ).stdout.strip()
        except CommandError as ce:
            NOT_A_GIT_REPO_ERROR = 128
            if ce.return_code == NOT_A_GIT_REPO_ERROR:
                self.log(
                    "git branch --show-current returned exit code "
                    f"{NOT_A_GIT_REPO_ERROR}. Current directory is not a git "
                    "repository."
                )
            # Silently gobble this error, it's mostly just informational
            return

        if not branch:
            # Per git branch --help: 'In detached HEAD state, nothing is
            # printed.'
            branch = "<detached HEAD>"

        try:
            commit = self.log_execute_command(
                [git_exe, "rev-parse", "HEAD"]
            ).stdout.strip()
        except CommandError:
            return

        self.log(f"Git branch: {branch}")
        self.log(f"Git commit: {commit}")

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

        class CustomFormatter(
            ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter
        ):
            pass

        descr = f"""\
        Configure {self.project_name}.

        On success, a directory {self.project_dir}/{{project arch name}} will
        be created, containing the configured build.

        Options listed below are handled directly by configure. Any options
        following a '--' are passed verbatim to CMake. For example:

        $ ./configure --with-cxx clang++ -- -DCMAKE_C_COMPILER='gcc'

        will set the C++ compiler to 'clang++' and the C compiler to 'gcc'.
        However, such manual intervention is rarely needed, and serves only
        as an escape hatch for as-of-yet unsupported arguments.
        """
        descr = textwrap.dedent(descr)

        parser = ArgumentParser(
            usage="%(prog)s [options...] [-- raw cmake options...]",
            description=descr,
            formatter_class=CustomFormatter,
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
        for action in parser._actions:  # noqa: SLF001
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
                    f'{arch_name} found in environment: "{environ[arch_name]}"'
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

    def _setup_arch_dir(self) -> None:
        r"""Ensure the creation and validity of the project arch directory.

        Raises
        ------
        RuntimeError
            If the arch directory exists but is not a directory.
        """
        arch_dir = self.project_arch_dir
        proj_name = self.project_name
        with_clean = self.cl_args.with_clean
        with_clean_val = with_clean.value
        if with_clean_val:
            self.log_warning(
                f"{dest_to_flag(with_clean.name)} specified, deleting "
                f"contents of {arch_dir}!"
            )

        if arch_dir.exists():
            self.log(f"{proj_name} arch exists: {arch_dir}")
            if not arch_dir.is_dir():
                msg = (
                    f"{proj_name} arch directory "
                    f"{arch_dir} already exists, but is not a "
                    "directory. Please delete move or delete this file "
                    "before re-running configure!"
                )
                raise RuntimeError(msg)
            if not with_clean_val:
                reconfigure_file = self._reconfigure.reconfigure_file
                if Path(sys.argv[0]).resolve() == reconfigure_file:
                    # The user is following our advice below and reconfiguring,
                    # so it's OK if the arch already exists.
                    self.log("User is reconfiguring, so no need to error out")
                    return

                cmake_cache = self.project_cmake_dir / "CMakeCache.txt"
                if not cmake_cache.exists():
                    # The cmake cache file doesn't exist, this would indicate
                    # that this is a new configuration. There is no need to
                    # error out because there are no prior effects that cmake
                    # could see.
                    self.log(
                        "CMake cache does not exist, so no need to error out "
                        "because for all intents and purposes, this is a "
                        "brand new configuration for CMake"
                    )
                    return

                force = self.cl_args.force
                if force.value:
                    self.log(
                        "User is forcing configuration, ignoring existing "
                        "arch dir"
                    )
                    return

                msg = (
                    f"{proj_name} arch directory {arch_dir} already exists and"
                    " would be overwritten by this configure command. If you:"
                    "\n"
                    "\n"
                    "  1. Meant to update an existing configuration, use "
                    f'{reconfigure_file.name} in place of "configure".'
                    "\n"
                    "  2. Meant to create a new configuration, re-run the "
                    "current configure command with "
                    f"--{self.project_arch_name}='some-other-name'."
                    "\n"
                    f"  3. Meant to redo the current arch ({arch_dir.name!r}) "
                    "from scratch, re-run configure with "
                    f"{dest_to_flag(with_clean.name)} option."
                    "\n"
                    "   4. Know what you are doing, and just want configure to"
                    " do as it is told, re-run the current configure command "
                    f"with {dest_to_flag(force.name)}"
                    "\n\n"
                )
                raise UnsatisfiableConfigurationError(msg)

            self.log("Deleting arch directory, then recreating it")
            proj_dir = self.project_dir
            if arch_dir == proj_dir or arch_dir in proj_dir.parents:
                # yes, this happened to me :(
                msg = (
                    f"Arch dir {arch_dir} is either a sub-path of or is the "
                    f"same as the project dir ({proj_dir}). Deleting the arch "
                    "dir would be catastrophic, probably this is a mistake!"
                )
                raise RuntimeError(msg)

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
        self._module_map = {
            type(conf_obj): idx for idx, conf_obj in enumerate(self._modules)
        }
        assert len(self._module_map) == len(self._modules), (
            "Duplicate modules!"
        )

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
            self._module_map[type(conf_obj)] = idx

        del self._topo_sorter

    def _get_package(self, req_package: type[Package]) -> Package:
        try:
            ret_idx = self._module_map[req_package]
        except KeyError as ke:
            raise ModuleNotFoundError(req_package) from ke

        return self._modules[ret_idx]  # should should never fail

    def _emit_summary(self) -> None:
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
        self.log_boxed(summary, title="Configuration Summary", align="left")
        install_mess = [
            "Please set the following:",
            "",
            f"export {self.project_arch_name}='{self.project_arch}'",
            f"export {self.project_dir_name}='{self.project_dir}'",
            "",
            "Then build libraries:",
            "$ make",
        ]

        from .package.packages.python import Python

        if self._get_package(Python).state.enabled():
            install_mess.extend(
                ("And install Python bindings:", "$ pip install .")
            )
        self.log_boxed(
            "\n".join(install_mess),
            title="Configuration Complete",
            align="left",
        )

    # Member variable access
    @property
    def argv(self) -> tuple[str, ...]:
        r"""Get the unparsed command-line arguments.

        Returns
        -------
        args : tuple[str, ...]
            The unparsed command-line arguments.
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
            msg = "Must call setup() first"
            raise WrongOrderError(msg)
        return self._cl_args

    @property
    def project_name(self) -> str:
        r"""Get the name of the current main project.

        Returns
        -------
        name : str
            The name of the current main project, e.g. "Legate".
        """
        return self._main_package.name

    @property
    def project_name_upper(self) -> str:
        r"""Get the name of the current main project in all caps,
        suitable for use as a variable.

        Returns
        -------
        name : str
            The name of the current main project, e.g. "LEGATE".
        """
        return self.project_name.replace(" ", "_").upper()

    @property
    def project_arch(self) -> str:
        r"""Get the current main project arch.

        Returns
        -------
        arch : str
          The arch name of the current main project, e.g. "arch-darwin-debug".
        """
        return self._main_package.arch_value

    @property
    def project_arch_name(self) -> str:
        r"""Get the current main project arch flag name.

        Returns
        -------
        flag_name : str
            The name of the current main project arch flag,
            e.g. "LEGATE_ARCH".
        """
        return self._main_package.arch_name

    @property
    def project_dir(self) -> Path:
        r"""Get the current main project root directory.

        Returns
        -------
        dir : Path
            The full path to the current project root directory, e.g.
            `/path/to/legate`.
        """
        return self._main_package.project_dir_value

    @property
    def project_src_dir(self) -> Path:
        r"""Get the current main project source directory.

        Returns
        -------
        dir : Path
            The full path to the current project source directory, e.g.
            `/path/to/legate/src`.
        """
        return self._main_package.project_src_dir

    @property
    def project_dir_name(self) -> str:
        r"""Get the name of the current main project root directory.

        Returns
        -------
        dir_name : Path
            The name of the current project root directory,
            e.g. "LEGATE_DIR".
        """
        return self._main_package.project_dir_name

    @property
    def project_arch_dir(self) -> Path:
        r"""Get the the current main project arch directory.

        Returns
        -------
        arch_dir : Path
            The full path to the current project arch directory,
            e.g. `/path/to/legate/arch-darwin-debug`.
        """
        return self.project_dir / self.project_arch

    @property
    def project_cmake_dir(self) -> Path:
        r"""Get the projects current cmake directory.

        Returns
        -------
        cmake_dir : Path
            The full path to the current project cmake directory.
            e.g. `/path/to/legate/arch-darwin-debug/cmake_build`.
        """
        return self.project_arch_dir / "cmake_build"

    @property
    def project_export_config_path(self) -> Path:
        r"""Get the projects export config file path.

        Returns
        -------
        export_path : Path
            The full path to the export config file containing all of the
            exported variables to be read back by aedifix once the cmake
            build completes, e.g.
            `/path/to/legate/arch-foo/cmake_build/aedifix_export_config.json`
        """
        return self.project_cmake_dir / "aedifix_export_config.json"

    @staticmethod
    def _sanitize_name(var: str | ConfigArgument) -> str:
        name: str
        if isinstance(var, ConfigArgument):
            if var.cmake_var is None:
                msg = f"CMake Variable for {var.name} is unset: {var}"
                raise ValueError(msg)
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

    # Logging
    def log(
        self,
        msg: str | list[str] | tuple[str, ...],
        *,
        tee: bool = False,
        caller_context: bool = True,
        keep: bool = False,
    ) -> None:
        r"""Append a message to the log.

        Parameters
        ----------
        msg : str | list[str] | tuple[str, ...]
            The message(s) to append to the log.
        tee : bool, False
            If True, output is printed to screen in addition to being appended
            to the on-disk log file. If False, output is only written to disk.
        caller_context : bool, True
            Whether to prepend the name of the function which called this
            function to `mess`.
        keep : bool, False
            Whether to make the message persist in live output.
        """
        verbose_mess = msg
        if caller_context:
            try:
                caller = get_calling_function()
            except ValueError:
                pass
            else:
                caller_name, _, _ = classify_callable(
                    caller, fully_qualify=False
                )
                match msg:
                    case str():
                        verbose_mess = f"{caller_name}(): {msg}"
                    case list() | tuple():
                        verbose_mess = [
                            f"{caller_name}(): {sub}" for sub in msg
                        ]
                    case _:
                        raise TypeError(msg)

        self._logger.log_file(verbose_mess)
        if tee:
            # See https://github.com/python/mypy/issues/18121 for why this
            # type-check is ignored
            self._logger.log_screen(msg, keep=keep)  # type: ignore[arg-type]

    def log_divider(self, *, tee: bool = False, keep: bool = True) -> None:
        r"""Append a dividing line to the logs.

        Parameters
        ----------
        tee : bool, False
           If True, output is printed to screen in addition to being appended
           to the on-disk log file. If False, output is only written to disk.
        keep : bool, True
           If ``tee`` is True, whether to persist the message in terminal
           output.
        """
        self._logger.log_divider(tee=tee, keep=keep)

    def log_boxed(
        self,
        message: str,
        *,
        title: str = "",
        title_style: str = "",
        align: AlignMethod = "center",
    ) -> None:
        r"""Log a message surrounded by a box.

        Parameters
        ----------
        message : str
            The message to log.
        title : str, ''
            An optional title for the box.
        title_style : str, ''
            Optional additional styling for the title.
        align : AlignMethod, 'center'
            How to align the text.
        """
        self._logger.log_boxed(
            message, title=title, title_style=title_style, align=align
        )

    def log_warning(self, message: str, *, title: str = "WARNING") -> None:
        r"""Log a warning to the log.

        Parameters
        ----------
        message : str
            The message to print.
        title : str, 'WARNING'
            The title to use for the box.
        """
        self._logger.log_warning(message, title=title)

    def log_error(self, message: str, *, title: str = "ERROR") -> None:
        r"""Log an error to the log.

        Parameters
        ----------
        message : str
            The message to print.
        title : str, 'ERROR'
            The title to use for the box.
        """
        self._logger.log_error(message, title=title)

    def log_execute_command(
        self, command: Sequence[_T], *, live: bool = False
    ) -> CompletedProcess[str]:
        r"""Execute a system command and return the output.

        Parameters
        ----------
        command : Sequence[T]
            The command list to execute.
        live : bool, False
            Whether to output the live output to screen as well (it is always
            updated continuously to the log file).

        Returns
        -------
        ret : CompletedProcess
            The completed process object.

        Raises
        ------
        RuntimeError
            If the command returns a non-zero errorcode
        """
        from rich.markup import escape

        def callback(stdout: str, stderr: str) -> None:
            if stdout := stdout.strip():
                if live:
                    stdout = escape(stdout)
                    lines = tuple(map(str.rstrip, stdout.splitlines()))
                    self.log(lines, caller_context=False, tee=True)
                else:
                    self.log(stdout, caller_context=False)
            if stderr := stderr.strip():
                self.log(f"STDERR:\n{stderr}", caller_context=False)

        self.log(f"Executing command: {' '.join(map(str, command))}")
        try:
            return subprocess_capture_output_live(
                command, callback=callback, check=True
            )
        except CommandError as ce:
            self.log(ce.summary)
            raise
        except Exception as e:
            self.log(str(e))
            raise

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
                if src_path.is_relative_to(parent):
                    return src_path.relative_to(parent)
            return src_path

        qual_name, src_path, lineno = classify_callable(fn)
        qual_path = pruned_path(src_path)
        self.log_divider()
        self.log(
            f"RUNNING: {qual_name}() ({qual_path}:{lineno})",
            tee=True,
            caller_context=False,
        )
        if docstr := inspect.getdoc(fn):
            self.log(f"  {docstr}\n", caller_context=False)
        else:
            # for a newline
            self.log("\n", caller_context=False)
        return fn(*args, **kwargs)

    # Meat and potatoes
    def require(self, package: Package, req_package: type[Package]) -> Package:
        r"""Indicate to the manager that `package` requires `req_package` to
        run before itself, and return a handle to the requested package.

        Parameters
        ----------
        package : Package
            The package that is requesting the dependency.
        req_package : type[Package]
            The class of the required package

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
            msg = (
                "Trying to require a module outside of setup_dependencies(), "
                "this is not allowed"
            )
            raise RuntimeError(msg) from ae

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
        self.log_execute_func(self._log_git_info)
        self._modules.extend(
            self.log_execute_func(packages.load_packages, self)
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
        self.log_execute_func(self._setup_arch_dir)
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
            self.project_src_dir,
            self.project_cmake_dir,
            extra_argv=self._extra_argv,
        )

        self.log_execute_func(self._config.finalize)
        self.log_execute_func(
            self._reconfigure.finalize,
            main_package_type=type(self._main_package),
            ephemeral_args=self._ephemeral_args,
            extra_argv=self._extra_argv,
        )
        self.log_execute_func(self._main_package.post_finalize)
        self.log_execute_func(self._emit_summary)
        self._logger.copy_log(self.project_arch_dir / "configure.log")

    def main(self) -> None:
        r"""Perform the main loop of the configuration."""
        with self._logger:
            self.setup()
            self.configure()
            self.finalize()
