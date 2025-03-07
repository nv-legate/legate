#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
from ctypes import CDLL, c_longlong, c_void_p
from pathlib import Path


def main(lib_path: Path) -> None:
    """Run the hello world example program.

    Parameters
    ----------
    lib_path : Path
        The path to the compiled library containing the hello world Legate
        task compiled from C++.
    """
    # Deliberately in main() so that the user can run --help on the program
    # without needing to build the .so (or setup legate) first.
    from legate import get_legate_runtime

    libhello_world = CDLL(lib_path)

    # Extract the task registration function from the shared object. It takes
    # as argument a pointer to the C++ Library object, and registers the task
    # with that library.
    hello_world_register_task = libhello_world.hello_world_register_task
    hello_world_register_task.argtypes = (c_void_p,)
    hello_world_register_task.restype = None

    # First, however, we create our library. As with almost all objects in
    # Legate, we create it through the Runtime.
    runtime = get_legate_runtime()
    lib = runtime.create_library("hello")

    # After this call returns, the Runtime will have all information required
    # for us to be able to create and submit our tasks. The registration call
    # will assign a globally unique Legion task ID, but also register the task
    # with its Library-local task ID.
    hello_world_register_task(lib.raw_handle)

    # Extract that function that gets our local task ID
    hello_world_local_task_id = libhello_world.hello_world_task_id
    hello_world_local_task_id.argtypes = ()
    hello_world_local_task_id.restype = c_longlong

    # Finally, we have the necessary tools to create an instance of our
    # task. We need to pass in the Library and the local task ID. The library
    # first translates our local task ID back into a global Legion ID, which
    # the runtime then uses to create the task instance.
    task = runtime.create_auto_task(lib, hello_world_local_task_id())
    # Hello World!
    runtime.submit(task)


if __name__ == "__main__":
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    match platform.system():
        case "Linux":
            so_ext = "so"
        case "Darwin":
            so_ext = "dylib"
        case "Windows":
            so_ext = "dll"
        case _:
            so_ext = "could_not_deduce"

    default_lib_path = (
        Path(__file__).parent / "build" / "lib" / f"libhello_world.{so_ext}"
    )

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-p",
        "--lib-path",
        required=not default_lib_path.exists(),
        type=Path,
        help="Path to the hello_world shared object",
        default=default_lib_path,
    )
    args = parser.parse_args()

    main(args.lib_path.resolve())
