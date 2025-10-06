..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. currentmodule:: legate.core

Arrays
======

``LogicalArray`` represents a set of one or more ``LogicalStore`` s that are partitioned in
a related way, to synthesize a more sophisticated data container. For example, an
integer-type ``LogicalStore`` can be combined with a boolean-type ``LogicalStore`` of the
same size to create a masked integer array.


.. autosummary::
   :toctree: generated/
   :template: class.rst

   LogicalArray
   StructLogicalArray
