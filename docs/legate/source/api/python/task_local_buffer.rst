..
  SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. currentmodule:: legate.core

Task-Local Buffers
==================

``TaskLocalBuffer`` is a buffer allocated during task execution for use as temporary
storage, or to later "bind" to an unbound ``PhysicalStore``. The lifetime of the buffer is
that of the task body. If the buffer was bound to a store, then the memory of the buffer
is transferred to the store on completion of the task. If the buffer was not bound to a
store, then the buffer is destroyed and the memory is reclaimed by the runtime.


.. autosummary::
   :toctree: generated/
   :template: class.rst

   TaskLocalBuffer
