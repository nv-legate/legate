#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path


def get_legate_dir() -> str:
    return str(Path(__file__).resolve().parents[1])


def main() -> None:
    print(get_legate_dir(), end="", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
