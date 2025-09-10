..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. _label_io:

.. currentmodule:: legate.core.experimental.io

===
I/O
===

Utilities for serializing and deserializing Legate stores.

HDF5
====

API
---

.. currentmodule:: legate.io.hdf5

.. autosummary::
   :toctree: ../generated/

   from_file
   from_file_batched
   to_file

KVikIO
======

.. currentmodule:: legate.core.experimental.io.file_handle

.. autosummary::
   :toctree: ../generated/

   from_file
   to_file

.. autosummary::
   :toctree: ../generated/
   :template: class.rst

   FileHandle
