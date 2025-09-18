# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

# make printers available to script
sys.path.append(str(Path(__file__).resolve().parent))

from legate_printers import sharedptr_printer, smallvec_printer, span_printer

import gdb
from gdb.printing import RegexpCollectionPrettyPrinter


def build_pretty_printer() -> RegexpCollectionPrettyPrinter:
    """Build the pretty printer for the Legate C++ library."""
    pp = RegexpCollectionPrettyPrinter("legate")
    smallvec_printer.register_printer(pp)
    sharedptr_printer.register_printer(pp)
    span_printer.register_printer(pp)
    return pp


gdb.pretty_printers.append(build_pretty_printer())
