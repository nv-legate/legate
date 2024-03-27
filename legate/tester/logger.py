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

"""Provide a basic logger that can scrub ANSI color codes.

"""
from __future__ import annotations

import re

# ref: https://stackoverflow.com/a/14693789
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class Log:
    def __init__(self) -> None:
        self._record: list[str] = []

    def __call__(self, *lines: str) -> tuple[int, int]:
        return self.record(*lines)

    def record(self, *lines: str) -> tuple[int, int]:
        if len(lines) == 1 and "\n" in lines[0]:
            lines = tuple(lines[0].split("\n"))

        start = len(self._record)
        for line in lines:
            self._record.append(line)
            print(line, flush=True)
        return (start, len(self._record))

    def clear(self) -> None:
        self._record = []

    def dump(
        self,
        *,
        start: int = 0,
        end: int | None = None,
        filter_ansi: bool = True,
    ) -> str:
        lines = self._record[start:end]

        if filter_ansi:
            full_text = _ANSI_ESCAPE.sub("", "\n".join(lines))
        else:
            full_text = "\n".join(lines)

        return full_text

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self._record)


LOG = Log()
