..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. currentmodule:: legate.core

Domains
=======

``DomainPoint`` and ``Domain`` allow the user to specify specific points or dense
rectangles within N-dimensional spaces. They are often used as task "indices" (to locate a
specific point task within a wider array of tasks), and to describe the overall launch
domain of tasks.

Being essentially a tuple and a pair of tuples, however, they are also more generally
useful.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   DomainPoint
   Domain
