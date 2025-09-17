..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. _ch_dim_ordering:

==================
Dimension Ordering
==================

This document describes the semantics of dimension ordering, a topic that is often confusing in the
presence of store transformations.

Definition
==========

A dimension ordering is a tuple of dimension indices, listed from the most rapidly changing
dimension to the least rapidingly changing one. For example, if we have a 3D store, the dimension
ordering ``(1, 0, 2)`` means that the second dimension of the store is the most rapidly changing one.

Oftentimes, we use the term "Fortran ordering", which corresponds to a dimension ordering where
dimension indices are listed in increasing order; i.e., for an N-D store, the Fortran ordering
is ``(0, 1, ..., N - 1)``. The term "C ordering" maps to the opposite ordering; i.e., for an N-D
store, the C ordering is ``(N - 1, ..., 1, 0)``. These colloquial terms have well-defined semantics
independent of the number of dimensions (Legate treats these two common cases in a special way).

Relationship to Instance Layout
-------------------------------

The ordering is translated into contiguity in the physical instance. Elements whose points are
adjacent along the most rapidly changing dimension appear contiguously in the instance. For a 3D
store ``st`` with a dimension ordering ``(1, 0, 2)``, elements ``st[i, j, k]`` and ``st[i, j + 1,
k]`` locate next to each other in the instance.

Store Transformations and Mapping to Legion
===========================================

The meaning of dimension ordering doesn't and shouldn't change even when the store is transformed.
This means that when a transform is applied to a store, the dimension ordering of the output store's
coordinate space should be inverted back into the input store's coordinate space given to Legion.

Here's a quick example: let's say we have a 2D store constructed by transposing another 2D store and
the mapper maps the store to an instance with Fortran ordering. In this case, the Legion level
instance should have the C ordering to look like a Fortran-ordered instance with respect to
the mapped store.

Now, let's look at a more complex example:

.. code-block:: python

   st1 = runtime.create_store(shape=(10, 11, 12), ...)
   st2 = st1.transpose((2, 0, 1))

If we want to map ``st2`` to an instance with C ordering, the dimension ordering ``(2, 1, 0)``
(which represents the C ordering) needs to be inverted to ``st1``'s coordinate space that Legion
sees.  Such an inversion is done by "chasing backward" the list of dimensions given to the transpose
call; i.e., as the point ``(i, j, k)`` in ``st2``'s coordinate space is mapped to ``(j, k, i)`` in
``st1``'s, the dimension ordering after inversion becomes ``(1, 0, 2)`` (which then is converted
into ``(LEGION_DIM_Y, LEGION_DIM_X, LEGION_DIM_Z)``).

Let's now see how this Legion dimension ordering guarantees the C ordering for ``st2``. Suppose we
have two elements ``st2[i, j, k]`` and ``st2[i, j, k + 1]`` that should appear consecutively in the
C layout. These two elements are aliases to elements ``st1[j, k, i]`` and ``st1[j, k + 1, i]``,
respectively. Because the Legion dimension ordering above chooses the second dimension
(``LEGION_DIM_Y``) to be the most rapidly changing dimension, the two elements would be adjacent to
each other in the instance.
