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
import platform
import subprocess
import sysconfig
from collections.abc import Callable, Sequence
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from sys import version_info
from typing import Any, TypeVar

_T = TypeVar("_T")


def subprocess_check_returncode(
    ret: CompletedProcess[_T],
) -> CompletedProcess[_T]:
    r"""Check the return code of a subprocess return value.

    Paramters
    ---------
    ret : CompletedProcess
      the return value of `subprocess.run()`

    Returns
    -------
    ret : CompletedProcess
      `ret` unchanged

    Raises
    ------
    RuntimeError
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
        raise RuntimeError(emess) from cpe
    return ret


def subprocess_capture_output(
    *args: Any, check: bool = True, **kwargs: Any
) -> CompletedProcess[str]:
    r"""Lightweight wrapper over `subprocess.run()`

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
    RuntimeError
        If `subprocess.run()` raises a `subprocess.CalledProcessError`, this
        routine converts it into a `RuntimeError` with the output attached.

    Notes
    -----
    Turns a `subprocess.CalledProcessError` into a `RuntimeError` with more
    diagnostics.
    """
    ret = subprocess.run(
        *args,
        capture_output=True,
        universal_newlines=True,
        check=False,
        **kwargs,
    )
    if check:
        ret = subprocess_check_returncode(ret)
    return ret


def copy_doc(source: Any) -> Callable[[_T], _T]:
    r"""Copy the docstring from one object to another

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
    r"""Determine the current Python version and the path to its
    shared library.

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
        raise FileNotFoundError("Could not auto-locate Python library") from ie

    if not py_lib_path.exists():
        raise RuntimeError(
            "Could not auto-locate Python library, "
            f"found library ({py_lib_path}) does not appear to exist"
        )
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
            raise ValueError(f"Argument '{arg}' must start with '-'")

    if not remove_args:
        return [a for a in argv]

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
    Deduplicates the arguments by keeping only the *last* occurance of each
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
