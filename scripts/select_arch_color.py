#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def select_arch_color(arch_value: str) -> str:
    r"""Chooses a random color based on the value of arch_value. This helps to
    visually distinguish between different arches when recompiling.

    Parameters
    ----------
    arch_value : str
        The arch value, or any string really.

    Returns
    -------
    str
        The chosen color.
    """
    from zlib import adler32

    colors = ("red", "green", "yellow", "blue", "magenta", "cyan", "normal")

    # Use zlib.adler32 instead of builtin hash() because the builtin hash() is
    # not reproducible. Running hash("the same string") on different
    # invocations of Python will yield different results every time. We don't
    # want that, because we want the same arch value to have the same color
    # each time:
    #
    # $ python3 -c 'print(hash("foo"))'
    # 673628788748047246
    # $ python3 -c 'print(hash("foo"))'
    # -6413379324416539761
    #
    # With adler32:
    # $ python3 -c 'import zlib; print(zlib.adler32("foo".encode()))'
    # 42074437
    # $ python3 -c 'import zlib; print(zlib.adler32("foo".encode()))'
    # 42074437
    return colors[adler32(arch_value.encode()) % len(colors)]


def main() -> None:
    from os import environ

    print(  # noqa: T201
        select_arch_color(environ.get("LEGATE_ARCH", "")), end=""
    )


if __name__ == "__main__":
    main()
