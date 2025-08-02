..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. currentmodule:: legate.core

Physical Store
==============

The ``PhysicalStore`` is the manifestation of (part of) a ``LogicalStore`` within a
task. Where ``LogicalStore`` can be used to abstractly describe the full dataset, with
additional transformations at the logical level (e.g. transposing, reshaping, slicing),
the ``PhysicalStore`` represents the actual piece of the data that a task is accessing.


.. autosummary::
   :toctree: generated/
   :template: class.rst

   PhysicalStore
