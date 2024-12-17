# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import sys
import traceback
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from .manager import ConfigurationManager
from .package.main_package import ON_ERROR_DEBUGGER_FLAG
from .util.exception import (
    CMakeConfigureError,
    UnsatisfiableConfigurationError,
)

if TYPE_CHECKING:
    from .package.main_package import MainPackage


SUCCESS: Final = 0
FAILURE: Final = 1


def _handle_generic_error(
    config: ConfigurationManager, message: str, title: str
) -> None:
    try:
        config.log_divider()
        config.log_warning(message, title=title)
        config.log_divider()
    except Exception as e:
        print(
            "Error printing error message from exception or "
            "printing the traceback:",
            str(e),
            flush=True,
        )
        print(title, flush=True)
        print(message, flush=True)


def _handle_exception(
    config: ConfigurationManager,
    excn_trace: str,
    title: str,
    excn_obj: Exception,
) -> None:
    try:
        config.log(excn_trace)
        excn_str = str(excn_obj)
        if not excn_str:
            excn_str = "[No Error Message Provided]"
        excn_str += (
            f", please see {config._logger.file_path} for "
            "additional details."
        )
        _handle_generic_error(config, excn_str, title)
    except Exception as e:
        print(
            "Error printing error message from exception or "
            "printing the traceback:",
            str(e),
            flush=True,
        )
        print(excn_trace, flush=True)


def _basic_configure_impl(
    argv: Sequence[str], MainPackageType: type[MainPackage]
) -> int:
    try:
        import ipdb as py_db  # type: ignore[import, unused-ignore]
    except ModuleNotFoundError:
        import pdb as py_db

    post_mortem = any(ON_ERROR_DEBUGGER_FLAG in arg for arg in argv)
    excn: Exception | None = None
    # If the following throws, then something is seriously beansed. Better to
    # eschew pretty-printing and just allow the entire exception to be printed.
    config = ConfigurationManager(argv, MainPackageType)
    try:
        try:
            config.main()
        except:  # noqa E722
            if post_mortem:
                py_db.post_mortem()
            raise
    except UnsatisfiableConfigurationError as e:
        title = "Configuration is not satisfiable"
        excn = e
    except CMakeConfigureError as e:
        title = "CMake configuration failed"
        excn = e
    except AssertionError as e:
        if excn_str := str(excn):
            title = (
                f"Assertion Error: {excn_str}, this indicates a configure bug!"
            )
        else:
            title = "CONFIGURATION CRASH"
        excn = e
    except KeyboardInterrupt:
        _handle_generic_error(
            config,
            message="Configuration was aborted by the user (recieved SIGINT)",
            title="Configuration Aborted",
        )
        return FAILURE
    except Exception as e:
        title = "CONFIGURATION CRASH"
        excn = e

    if excn is not None:
        trace = "".join(traceback.format_exception(excn, chain=True))
        _handle_exception(config, trace, title, excn)
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
        try:
            sys.stdout.flush()
        except:  # noqa E722
            pass
        try:
            sys.stderr.flush()
        except:  # noqa E722
            pass
