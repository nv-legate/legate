# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
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
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Literal

sys.path.append(str(Path(__file__).parents[2]))

from get_legate_dir import get_legate_dir  # type: ignore[import-not-found]

from .bump_cmake_versions import get_cmakelists_version

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

        cmakelists = self.legate_dir / "src" / "CMakelists.txt"
        self.vprint(f"Opening {cmakelists}")
        version, _ = get_cmakelists_version(
            cmakelists, cmakelists.read_text().splitlines()
        )

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
