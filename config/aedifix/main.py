# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import traceback
from argparse import ArgumentError
from contextlib import suppress
from typing import TYPE_CHECKING, Final

from .manager import ConfigurationManager
from .package.main_package import ON_ERROR_DEBUGGER_FLAG
from .util.exception import (
    CMakeConfigureError,
    UnsatisfiableConfigurationError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .package.main_package import MainPackage


SUCCESS: Final = 0
FAILURE: Final = 1


def _handle_generic_error(
    config: ConfigurationManager, message: str, title: str
) -> None:
    try:
        config.log_divider()
        config.log_error(message, title=title)
        config.log_divider()
    except Exception as e:
        print(  # noqa: T201
            "Error printing error message from exception or "
            "printing the traceback:",
            str(e),
            flush=True,
        )
        print(title, flush=True)  # noqa: T201
        print(message, flush=True)  # noqa: T201


def _handle_exception(
    config: ConfigurationManager, title: str, excn_obj: Exception
) -> None:
    trace = "".join(traceback.format_exception(excn_obj, chain=True))
    excn_str = str(excn_obj)
    if not excn_str:
        excn_str = "[No Error Message Provided]"

    log_path: str | Path
    try:
        log_path = config._logger.file_path  # noqa: SLF001
    except Exception:
        log_path = "configure.log"

    excn_str += f", please see {log_path} for additional details."
    try:
        config.log(trace)
        _handle_generic_error(config, excn_str, title)
    except Exception as e:
        print(  # noqa: T201
            "Error printing error message from exception or "
            "printing the traceback:",
            str(e),
            flush=True,
        )
        print(trace, flush=True)  # noqa: T201


def _basic_configure_impl(
    argv: Sequence[str], MainPackageType: type[MainPackage]
) -> int:
    try:
        import ipdb as py_db  # type: ignore[import, unused-ignore] # noqa: T100
    except ModuleNotFoundError:
        import pdb as py_db  # noqa: T100

    post_mortem = any(ON_ERROR_DEBUGGER_FLAG in arg for arg in argv)
    excn: Exception | None = None
    # If the following throws, then something is seriously beansed. Better to
    # eschew pretty-printing and just allow the entire exception to be printed.
    config = ConfigurationManager(argv, MainPackageType)
    try:
        try:
            config.main()
        except:
            if post_mortem:
                py_db.post_mortem()
            raise
    except UnsatisfiableConfigurationError as e:
        title = "Configuration is not satisfiable"
        excn = e
    except CMakeConfigureError as e:
        title = "CMake configuration failed"
        excn = e
    except KeyboardInterrupt:
        _handle_generic_error(
            config,
            message="Configuration was aborted by the user (received SIGINT)",
            title="Configuration Aborted",
        )
        return FAILURE
    except ArgumentError as e:
        title = "Invalid Option"
        excn = e
    except Exception as e:
        title = "CONFIGURATION CRASH"
        excn = e

    if excn is not None:
        _handle_exception(config, title, excn)
        return FAILURE
    return SUCCESS


def basic_configure(
    argv: Sequence[str], MainPackageType: type[MainPackage]
) -> int:
    r"""Run a basic configuration.

    Parameters
    ----------
    argv : Sequence[str]
        The command line arguments to configure with.
    MainPackageType : type[MainPackage]
        The type of the main package for which to configure.

    Returns
    -------
    ret : int
        The return code to return to the calling shell. On success, returns
        `SUCCESS`, on failure, returns `FAILURE`.
    """
    try:
        return _basic_configure_impl(argv, MainPackageType)
    finally:
        # Flush both streams on end. This is needed because if there is an
        # error in CI, the internal buffering won't properly flush the error
        # message and we get garbled output.
        with suppress(Exception):
            sys.stdout.flush()
        with suppress(Exception):
            sys.stderr.flush()
