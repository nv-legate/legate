# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


class BaseError(Exception):
    r"""Base exception."""


class UnsatisfiableConfigurationError(BaseError):
    r"""An error as a result of user configuration that cannot be satisfied.
    For example, the user has requested X, but it does not work. Or the user
    has told us to look for Y in Z, but we did not find it there.
    """


class CMakeConfigureError(BaseError):
    r"""An error as a result of CMake failing."""


class LengthError(BaseError):
    r"""An exception to signify an object that is not of the right length."""


class WrongOrderError(BaseError):
    r"""An error raised when an operation is performed in the wrong order, e.g.
    accessing an object before it has been setup, or retrieving a resource
    before it has been registered.
    """


class CommandError(BaseError):
    r"""An error raised when an external command returns an error."""

    def __init__(
        self,
        return_code: int,
        stdout: str,
        stderr: str,
        summary: str | None = None,
    ) -> None:
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        if summary is None:
            summary = self._make_summary(return_code, stdout, stderr)
        self.summary = summary

    @staticmethod
    def _make_summary(return_code: int, stdout: str, stderr: str) -> str:
        lines = (
            f"Subprocess error, returned exit-code: {return_code}",
            "stdout:",
            f"{stdout}",
            "stderr:",
            f"{stderr}",
        )
        return "\n".join(lines)
