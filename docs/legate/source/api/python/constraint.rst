..
  SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. currentmodule:: legate.core

Constraints
===========

``Constraint`` represents a specific partitioning constraint imposed on an argument to a
task. For example, the ``broadcast()`` constraint requires that all leaf tasks get a
complete copy of the argument, instead of a subset. ``align()`` on the other hand mandates
that two arguments must be partitioned in the same way (their pieces align each other).


.. autosummary::
   :toctree: generated/
   :template: class.rst

   Constraint
