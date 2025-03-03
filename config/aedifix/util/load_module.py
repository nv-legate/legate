# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pydoc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def load_module_from_path(path: Path | str) -> ModuleType:
    r"""Load a module given an absolute path.

    Parameters
    ----------
    path : Path | str
        The absolute path to the python module to load.

    Returns
    -------
    mod : ModuleType
        The loaded module.

    Raises
    ------
    ImportError
        If the module failed to import.
    """
    # I tried every given solution at:
    #
    # https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
    #
    # and this was -- unironically -- the most complete answer. All of the
    # others would produce usable modules, but then some thing or another would
    # be "off" about them. For example, inspect.getmodule() would fail to find
    # the module (i.e. return None), or sometimes they were missing __name__
    # and __package__ attributes (i.e. empty).
    try:
        return pydoc.importfile(str(path))
    except pydoc.ErrorDuringImport as edi:
        msg = f"Failed to import {path}"
        raise ImportError(msg) from edi
