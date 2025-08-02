..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. currentmodule:: legate.core

Auto Tasks
==========

``AutoTask`` represents a unit of work that can be split in flexible ways. The "auto" in
``AutoTask`` refers to the fact that the task launch is automatically parallelized by the
runtime. On launch, the runtime will determine how many leaf tasks to instantiate, which
pieces of the arguments each leaf task will access, and what processor each leaf task will
be placed on.

See ``ManualTask`` for tasks which need to be manually parallelized.


.. autosummary::
   :toctree: generated/
   :template: class.rst

   AutoTask
