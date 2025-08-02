..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. currentmodule:: legate.core

Types
=====

``Type`` is the fundamental unit of the type system in Legate. Compound types may be
constructed using ``StrucType``, with special support provided for array-like types in
``FixedArrayType``.

These types are usually used in the construction of the higher-level container classes
(such as ``LogicalArray``, ``LogicalStore``, or ``Scalar``).


.. autosummary::
   :toctree: generated/
   :template: class.rst

   Type
   StructType
   FixedArrayType
