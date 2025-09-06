..
  SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

=============
Code Linting
=============

Clang-Tidy
==========

Running Tidy
------------

Use the aggregate ``make tidy`` target to lint the C++ codebase.

1) Configure the repo (tidy targets are always generated):

.. code-block:: sh

   ./configure --cmake-generator=Ninja

2) Run clang-tidy across all sources:

.. code-block:: sh

   make tidy -j $(nproc)

Running Tidy With CUDA (Optional)
---------------------------------

When you also want to lint CUDA kernels (``.cu`` files):

1) Configure with CUDA and clang-as-CUDA so tidy analyzes ``.cu`` files with a
   clang-compatible toolchain:

.. code-block:: sh

   ./configure --with-cuda \
               --with-cuda-dir="${CONDA_PREFIX}" \
               --with-fake-fatbins-for-tidy \
               --with-cudac=clang++ \
               --CUDAFLAGS "-O3" \
               --cuda-arch=90

2) Run all tidy targets (C++ and CUDA) via the aggregate target:

.. code-block:: sh

   make tidy -j $(nproc)

CUDA options
------------

- ``--with-cuda``: enable CUDA toolchain so ``.cu`` files are analyzed.
- ``--with-cuda-dir=...``: point to the CUDA toolkit when using a conda toolchain.
- ``--with-cudac=clang++``: use clang for CUDA (required for CUDA tidy).
- ``--with-fake-fatbins-for-tidy``: generate stub fatbins so host code that
  embeds fatbins can compile under tidy without building fatbins/Legion.
- ``--CUDAFLAGS "-O3"``: minimal CUDA flags for tidy's compile database.
- ``--cuda-arch=...``: target GPU architecture for device-only analysis. clang-tidy
  supports only a single architecture at a time -- pass one (e.g., ``90``).

NOLINT guidance
---------------

- Prefer narrow, inline suppressions (``// NOLINT(<check>)``) at the exact
  diagnostic site.
- Inside macro definitions with line continuations, use block comments
  (``/* NOLINT(...) */``) before the trailing ``\`` or wrap the macro with a
  narrow ``NOLINTBEGIN/END`` for ``bugprone-macro-parentheses``.
