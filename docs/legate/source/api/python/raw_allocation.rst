..
  SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. currentmodule:: legate.core

Raw Allocations
===============

``RawAllocation`` is a handle to the raw memory backing a ``PhysicalStore`` (its base pointer,
size, and memory kind). It can be obtained from ``PhysicalStore.get_raw_allocation()``, and is
also used to create Legate stores from memory external to Legate.


.. autosummary::
   :toctree: generated/
   :template: class.rst

   RawAllocation
