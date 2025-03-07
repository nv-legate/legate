# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import enum
import shlex
import platform
import sysconfig
import subprocess
from pathlib import Path
from signal import SIGINT
from subprocess import (
    PIPE,
    STDOUT,
    CalledProcessError,
    CompletedProcess,
    Popen,
    TimeoutExpired,
)
from sys import version_info
from typing import TYPE_CHECKING, Any, Final, TypeVar

from .exception import CommandError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from ..base import Configurable

_T = TypeVar("_T")


def subprocess_check_returncode(
    ret: CompletedProcess[_T],
) -> CompletedProcess[_T]:
    r"""Check the return code of a subprocess return value.

    Parameters
    ----------
    ret : CompletedProcess
      the return value of `subprocess.run()`

    Returns
    -------
    ret : CompletedProcess
      `ret` unchanged

    Raises
    ------
    CommandError
      If `ret.returncode` is nonzero
    """
    try:
        ret.check_returncode()
    except CalledProcessError as cpe:
        emess = "\n".join(
            [
                "Subprocess error:",
                "stdout:",
                f"{cpe.stdout}",
                "stderr:",
                f"{cpe.stderr}",
                f"{cpe}",
            ]
        )
        raise CommandError(
            return_code=cpe.returncode,
            stdout=cpe.stdout,
            stderr=cpe.stderr,
            summary=emess,
        ) from cpe
    return ret


def subprocess_capture_output(
    *args: Any, check: bool = True, **kwargs: Any
) -> CompletedProcess[str]:
    r"""Lightweight wrapper over `subprocess.run()`.

    Parameters
    ----------
    *args : Any
        Arguments to `subprocess.run()`.
    check : bool, True
        Whether to check the return code.
    **kwargs : Any
        Keyword arguments to `subprocess.run()`

    Returns
    -------
    ret : CompletedProcess
        The return value of `subprocess.run()`.

    Raises
    ------
    CommandError
        If `subprocess.run()` raises a `subprocess.CalledProcessError`, this
        routine converts it into a `CommandError` with the output attached.

    Notes
    -----
    Turns a `subprocess.CalledProcessError` into a `CommandError` with more
    diagnostics.
    """
    ret = subprocess.run(
        *args, capture_output=True, text=True, check=False, **kwargs
    )
    if check:
        ret = subprocess_check_returncode(ret)
    return ret


def _normalize_output(output: bytes | None | str) -> str:
    match output:
        case None:
            return ""
        case str():
            return output
        case bytes():
            return output.decode()


def subprocess_capture_output_live_impl(
    callback: Callable[[str, str], None], *args: Any, **kwargs: Any
) -> CompletedProcess[str]:
    kwargs.setdefault("stdout", PIPE)
    kwargs.setdefault("stderr", STDOUT)
    kwargs["universal_newlines"] = True
    kwargs["text"] = True

    timeout = kwargs.pop("timeout", 1)

    total_stdout_len = 0
    total_stderr_len = 0
    stdout = ""
    stderr = ""

    with Popen(*args, **kwargs) as process:
        done = False
        while not done:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except TimeoutExpired as te_exn:
                stdout = _normalize_output(te_exn.stdout)
                stderr = _normalize_output(te_exn.stderr)
            except KeyboardInterrupt:
                process.send_signal(SIGINT)
                raise
            except:
                process.kill()
                raise
            else:
                # Don't break, instead wait for end of loop, in case stdout
                # and/or stderr were updated.
                done = True
                stderr = _normalize_output(stderr)
                stdout = _normalize_output(stdout)

            # The streams will always contain the sum-total output of the
            # subprocess call, so we need to strip the stuff we've already
            # "seen" from the output before sending it to the callback.
            new_stdout = stdout[total_stdout_len:]
            new_stderr = stderr[total_stderr_len:]
            if new_stdout or new_stderr:
                # Now we replace the complete stdout
                total_stdout_len = len(stdout)
                total_stderr_len = len(stderr)
                callback(new_stdout, new_stderr)

        retcode = process.poll()
        if retcode is None:
            retcode = 0

    return CompletedProcess(process.args, retcode, stdout, stderr)


def subprocess_capture_output_live(
    *args: Any,
    callback: Callable[[str, str], None] | None = None,
    check: bool = True,
    **kwargs: Any,
) -> CompletedProcess[str]:
    r"""Execute a subprocess call with a live callback.

    Parameters
    ----------
    *args : Any
        Positional arguments to Popen.
    callback : Callable[[str, str], None], optional
        The callback to intermittently execute.
    check : bool, True
        Whether to check the returncode.
    **kwargs : Any
        Keyword arguments to Popen.

    Returns
    -------
    ret : CompletedProcess
        The object representing the subprocess results.

    Raises
    ------
    CommandError
        If `subprocess.run()` raises a `subprocess.CalledProcessError`, this
        routine converts it into a `CommandError` with the output attached.

    Notes
    -----
    The utility of this routine is to be able to monitor the output of the
    running subprocess in real time. This is done via the callback argument,
    which takes as arguments the stdout and stderr of the executing process.

    If callback is None, this routine is identical to
    subprocess_capture_output().
    """
    if callback is None:
        return subprocess_capture_output(*args, check=check, **kwargs)

    ret = subprocess_capture_output_live_impl(callback, *args, **kwargs)
    if check:
        ret = subprocess_check_returncode(ret)
    return ret


def copy_doc(source: Any) -> Callable[[_T], _T]:
    r"""Copy the docstring from one object to another.

    Parameters
    ----------
    source : Any
        The object to copy the docstring from

    Returns
    -------
    wrapper : Callable[[T], T]
        A wrapper which takes a target object and sets it docstring to that
        of `source`
    """

    def wrapper(target: _T) -> _T:
        if (sdoc := getattr(source, "__doc__", None)) is not None:
            target.__doc__ = sdoc
        return target

    return wrapper


class ValueProvenance(enum.Enum):
    COMMAND_LINE = enum.auto()
    ENVIRONMENT = enum.auto()
    GENERATED = enum.auto()


def find_active_python_version_and_path() -> tuple[str, Path]:
    r"""Determine the current Python version and the path to its shared
    library.

    Returns
    -------
    version : str
        The current Python version as a string.
    lib_path : Path
        The full path to the python shared library.

    Raises
    ------
    FileNotFoundError
        If the python shared library could not be located.
    """
    # Launching a sub-process to do this in a general way seems hard
    version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    cv = sysconfig.get_config_vars()
    # Homebrew or pkg mgr installations may give bad values for LDLIBRARY.
    # Uses a fallback default path in case LDLIBRARY fails.
    default_libname = f"libpython{cv['LDVERSION']}.a"
    libdirs = [str(cv["LIBDIR"]), str(cv["LIBPL"])]
    libnames = [str(cv["LDLIBRARY"]), default_libname]
    paths = [
        libdir / libname
        for libdir in map(Path, libdirs)
        for libname in libnames
    ]
    # ensure that static libraries are replaced with the dynamic version
    shlib_suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    paths = [p.with_suffix(shlib_suffix) for p in paths]
    paths = [p for p in paths if p.is_file()]
    try:
        py_lib_path = paths[0]
    except IndexError as ie:
        msg = "Could not auto-locate Python library"
        raise FileNotFoundError(msg) from ie

    if not py_lib_path.exists():
        msg = (
            "Could not auto-locate Python library, "
            f"found library ({py_lib_path}) does not appear to exist"
        )
        raise RuntimeError(msg)
    return version, py_lib_path


def prune_command_line_args(
    argv: Sequence[str], remove_args: set[str]
) -> list[str]:
    r"""Remove a set of command line arguments from argv.

    Parameters
    ----------
    argv : Sequence[str]
        The command line arguments to prune.
    remove_args : set[str]
        The arguments to remove.

    Returns
    -------
    argv : list[str]
        The pruned command line arguments.

    Raises
    ------
    ValueError
        If any of the arguments in `remove_args` do not start with '-'.
    """
    for arg in remove_args:
        if not arg.startswith("-"):
            msg = f"Argument '{arg}' must start with '-'"
            raise ValueError(msg)

    if not remove_args:
        return list(argv)

    idx = 0
    cl_args = []
    nargs = len(argv)
    while idx < nargs:
        arg = argv[idx]
        idx += 1
        if arg.split("=")[0] in remove_args:
            # we intend to skip this argument
            # if "=" in arg:
            #     # have --foo=x, can bail now
            #     continue
            # have
            #
            # --foo[=something] <maybe some stuff...> --bar
            #
            # So we want to iterate through array until we find the
            # next flag.
            while idx < nargs:
                if (arg := argv[idx]).startswith("-"):
                    # found flag
                    break
                idx += 1
            continue
        cl_args.append(arg)
    return cl_args


def deduplicate_command_line_args(argv: Sequence[str]) -> list[str]:
    r"""Deduplicate a set of command-line arguments.

    Parameters
    ----------
    argv : Sequence[str]
        The command line arguments to deduplicate.

    Returns
    -------
    argv : list[str]
        The deduplicated command line arguments.

    Notes
    -----
    Deduplicates the arguments by keeping only the *last* occurrence of each
    command line flag and its values.
    """
    # A dummy name that is used only in case the first arguments are
    # positional. Currently configure does not actually have any such arguments
    # (and, in fact, this function does not handle any remaining positional
    # arguments correctly), but good to be forward-looking.
    arg_name = "===POSITIONAL=FIRST=ARGUMENTS==="
    last_seen: dict[str, list[str]] = {arg_name: []}
    for arg in argv:
        if arg.startswith("-"):
            # --foo=bar
            # -> arg_name = --foo
            # -> *rest = bar
            arg_name, *rest = arg.split("=")
            # Clobbering the old last_seen[arg_name] is intentional
            last_seen[arg_name] = []
        last_seen[arg_name].append(arg)

    return [v for values in last_seen.values() for v in values]


def flag_to_dest(flag: str) -> str:
    r"""Convert a command-line flag to a 'dest' argument usable in an
    ArgumentParser.

    Parameters
    ----------
    flag : str
        The flag to convert.

    Returns
    -------
    dest : str
        The flag in 'dest' form.
    """
    return flag.lstrip("-").casefold().replace("-", "_")


def dest_to_flag(dest: str) -> str:
    r"""Convert a 'dest' argument usable in an ArgumentParser to the
    corresponding command-line flag.

    Parameters
    ----------
    dest : str
        The argument to convert.

    Returns
    -------
    flag : str
        The command-line flag.
    """
    return "--" + dest.replace("_", "-")


def partition_argv(argv: Iterable[str]) -> tuple[list[str], list[str]]:
    r"""Split a command-line list of arguments into 2.

    Parameters
    ----------
    argv : Iterable[str]
        The original argv to split.

    Returns
    -------
    main_argv : list[str]
        The argument before the first '--'
    rest_argv : list[str]
        The arguments after the first '--'
    """
    main_argv = []
    rest_argv = []
    found_sep = False
    for arg in argv:
        if arg.strip() == "--":
            found_sep = True
            continue

        if found_sep:
            rest_argv.append(arg)
        else:
            main_argv.append(arg)

    return main_argv, rest_argv


CMAKE_TEMPLATES_DIR: Final = Path(__file__).resolve().parents[1] / "templates"
CMAKE_CONFIGURE_FILE: Final = CMAKE_TEMPLATES_DIR / "configure_file.cmake"

assert CMAKE_CONFIGURE_FILE.exists(), (
    f"Cmake configure file {CMAKE_CONFIGURE_FILE} does not exist"
)
assert CMAKE_CONFIGURE_FILE.is_file(), (
    f"Cmake configure file {CMAKE_CONFIGURE_FILE} is not a file"
)


def cmake_configure_file(
    obj: Configurable, src_file: Path, dest_file: Path, defs: dict[str, Any]
) -> None:
    r"""Configure a file using CMake's configure_file().

    Parameters
    ----------
    obj : Configurable
        The configurable to use to launch the cmake command.
    src_file : Path
        The input file (i.e. the "template" file) to configure.
    dest_file : Path
        The destination file, where the output should be written.
    defs : dict[str, Any]
        A mapping of variable names to values to be replaced. I.e. ``@key@``
        will be replaced by ``value``.
    """
    cmake_exe = obj.manager.get_cmake_variable("CMAKE_COMMAND")
    base_cmd = [
        cmake_exe,
        f"-DAEDIFIX_CONFIGURE_FILE_SRC={src_file}",
        f"-DAEDIFIX_CONFIGURE_FILE_DEST={dest_file}",
    ]
    defs_cmd = [
        f"-D{key}={shlex.quote(str(value))}" for key, value in defs.items()
    ]

    if unhandled_subs := {
        var_name
        for var_name in re.findall(r"@([\w_]+)@", src_file.read_text())
        if var_name not in defs
    }:
        msg = (
            f"Substitution(s) {unhandled_subs} for {src_file} not found in "
            f"defs {defs.keys()}"
        )
        raise ValueError(msg)

    cmd = base_cmd + defs_cmd + ["-P", CMAKE_CONFIGURE_FILE]
    obj.log_execute_command(cmd)
