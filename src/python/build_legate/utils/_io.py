# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import os
import sys
import time
import shutil
import platform
from contextlib import contextmanager
from pathlib import Path
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Any, Final

import rich

from ._legate_config import get_legate_config

if TYPE_CHECKING:
    from collections.abc import Generator

    from ._types import BuildKind

BANNER_LEN: Final = 80
ERROR_BANNER: Final = f"XXX== {BANNER_LEN * '='} ==XXX"


def rich_print(*args: Any, **kwargs: Any) -> None:
    rich.print(*args, **kwargs)


def vprint(*args: Any, **kwargs: Any) -> None:
    rich_print("[bold]x-- [yellow]legate:setup.py:[/]", *args, **kwargs)


def warning_print(text: str, **kwargs: Any) -> None:
    kwargs.setdefault("flush", True)
    rich_print(
        "\n"
        f"[red bold]{ERROR_BANNER}[/]\n"
        f"[red]WARNING:[/] {text}\n"
        f"[red bold]{ERROR_BANNER}",
        **kwargs,
    )


@contextmanager
def Tee(log_file: Path) -> Generator[Path, None, None]:
    r"""Set up tee-ing of stdout and stderr to the log file.

    Parameters
    ----------
    log_file : Path
        The log file to tee stdout and stderr file streams into.

    Yields
    ------
    Path
        The path to the log file.

    Notes
    -----
    On systems that support it, this routine will use the standard ``tee``
    executable to perform the stream indirection. This ensures that any output
    by spawned subprocesses is also captured.

    For systems which do not support ``tee``, built-in contextlib redirection
    is used, which has no effect on the output of spawned subprocesses.
    """
    stdout = sys.stdout
    stderr = sys.stderr
    stdout.flush()
    stderr.flush()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w"):
        pass  # clear the file
    if tee_exec := shutil.which("tee"):
        # Note: this Popen must not be closed by python! Since we connect our
        # own stdout to it, the current process must first exit before it can
        # be closed.
        tee = Popen([tee_exec, log_file], stdin=PIPE)
        assert tee.stdin is not None  # pacify mypy
        tee_stdin = tee.stdin.fileno()
        # Cause tee's stdin to get a copy of our stdout/stderr (as well as that
        # of any child processes we spawn)
        os.dup2(tee_stdin, stdout.fileno())
        os.dup2(tee_stdin, stderr.fileno())
        yield log_file
        stdout.flush()
        stderr.flush()
    else:
        from contextlib import redirect_stderr, redirect_stdout

        with (
            log_file.open("w") as fd,
            redirect_stdout(fd),
            redirect_stderr(fd),
        ):
            yield log_file


@contextmanager
def BuildLog(build_kind: BuildKind) -> Generator[None, None, None]:
    r"""Initialize and create a build log for the configuration run.

    Parameters
    ----------
    build_kind : BuildKind
        The build type. This will also be the name of the logfile.

    Yields
    ------
    Generator[None]
        A generator, after which the log is automatically closed and
        flushed.
    """
    BANNER = f"xxx-- {BANNER_LEN * '='} --xxx"
    log_file = Path("exception_raised_before_log_init")
    try:
        with Tee(
            get_legate_config().SKBUILD_BUILD_DIR / f"{build_kind}.log"
        ) as log_file:
            rich_print(f"[bold yellow]{BANNER}")
            rn = time.strftime("%a, %d %b %Y %H:%M:%S %z")
            rich_print(f"[bold]Starting {build_kind!r} run at {rn}")
            rich_print("[bold]System info:")
            rich_print(platform.uname())
            rich_print(f"[bold]Writing log to: {log_file}")
            rich_print(f"[bold yellow]{BANNER}")

            yield
    except Exception as exn:
        mess = f"ERROR: {exn}\n\nPlease see {log_file} for further details"
        warning_print(mess)
        raise
