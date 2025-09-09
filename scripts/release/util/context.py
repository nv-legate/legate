# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Literal

sys.path.append(str(Path(__file__).parents[2]))

from get_legate_dir import get_legate_dir  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Sequence
    from subprocess import CompletedProcess
    from typing import Any


Mode = Literal["cut-branch", "post-cut"]


class Context:
    def __init__(self, args: Namespace) -> None:
        self._legate_dir = Path(get_legate_dir())
        self._verbose = args.verbose
        self._dry_run = args.dry_run
        self._version_after_this = args.version_after_this
        self._mode = args.mode

        # Determine the version being released from the top-level VERSION file
        version_file = self.legate_dir / "VERSION"
        self.vprint(f"Opening {version_file}")
        version = version_file.read_text().strip()
        # Keep just major.minor (e.g. 25.09 from 25.09.00)
        version = ".".join(version.split(".")[:2])
        self._version_being_released = version

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def version_being_released(self) -> str:
        return self._version_being_released

    @property
    def version_after_this(self) -> str:
        return self._version_after_this

    @staticmethod
    def to_full_version(version: str, *, extra_zeros: bool = False) -> str:
        null_ver = "0"
        if extra_zeros:
            null_ver += "0"

        tmp = version.split(".")
        MAX_VER_LEN = 3
        while len(tmp) < MAX_VER_LEN:
            tmp.append(null_ver)
        return ".".join(tmp)

    @property
    def legate_dir(self) -> Path:
        return self._legate_dir

    @property
    def mode(self) -> Mode:
        return self._mode

    def vprint(self, *args: Any, **kwargs: Any) -> None:
        if self.verbose:
            self.print(*args, **kwargs)

    def print(self, *args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)  # noqa: T201

    def run_cmd(
        self,
        cmd: Sequence[str],
        *args: Any,
        capture_output: bool = True,
        text: bool = True,
        check: bool = True,
        **kwargs: Any,
    ) -> CompletedProcess[str]:
        self.vprint(" ".join(cmd))
        return subprocess.run(
            cmd,
            *args,
            capture_output=capture_output,
            text=text,
            check=check,
            **kwargs,
        )
