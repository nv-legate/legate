.. SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: LicenseRef-NvidiaProprietary

   NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
   property and proprietary rights in and to this material, related
   documentation and any modifications thereto. Any use, reproduction,
   disclosure or distribution of this material and related documentation
   without an express license agreement from NVIDIA CORPORATION or
   its affiliates is strictly prohibited.

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

.. toctree::
  :maxdepth: 1

  versions

Indices and tables
------------------

* :ref:`genindex`
