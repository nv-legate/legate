.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Welcome to Legate.STL
=====================

Legate.STL is a C++ library that provides an STL-like iterface for
working with the Legate distributed runtime.

It provides:

* an STL container-like wrapper for a Legate logical store,

* views of physical storage as ``std::mdspan`` objects,

* range-like adaptors for projecting, filtering, and slicing logical
  and physical storage, and

* a set of STL-like algorithms for operating on ranges of slices.

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  user/index
  reference/index

Indices and tables
------------------

* :ref:`genindex`
