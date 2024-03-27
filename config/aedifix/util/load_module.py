#!/usr/bin/env python3
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
from __future__ import annotations

import pydoc
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
        raise ImportError(f"Failed to import {path}") from edi
