.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0


Installation
============

Legate.STL is available as part of the Legate project. To install
Legate follow the instructions in the section :ref:`how-do-i-install-legate`.

The default package contains GPU support and is compatible with CUDA Developer
Kit version 12.0 and greater.

Building
--------

Legate.STL is a header-only library. To use it,
``#include <legate/experimental/stl.hpp>``. The library's program
entities reside in the ``legate::experimental::stl`` namespace.

All examples in this documentation assume an initial
``namespace stl = legate::experimental::stl;`` directive.
