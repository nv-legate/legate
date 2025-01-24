.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: LicenseRef-NvidiaProprietary

   NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
   property and proprietary rights in and to this material, related
   documentation and any modifications thereto. Any use, reproduction,
   disclosure or distribution of this material and related documentation
   without an express license agreement from NVIDIA CORPORATION or
   its affiliates is strictly prohibited.

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
