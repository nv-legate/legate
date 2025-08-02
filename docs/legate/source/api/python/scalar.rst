..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. currentmodule:: legate.core

Scalar
======

``Scalar`` represents a single, immutable value within a task. These are usually small or
trivially copyable types. Unlike ``LogicalStore`` or ``LogicalArray`` which are
partitioned and split across tasks for a particular task launch, ``Scalar`` s are always
copied across all instances.


.. autosummary::
   :toctree: generated/
   :template: class.rst

   Scalar
