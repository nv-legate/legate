..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

=========================
Debugging Legate Programs
=========================

Pretty Printing
===============

To easily inspect some Legate objects, Legate provides pretty printers
for GDB and LLDB. The source code for these printers is in the ``share/legate/gdb``
and ``share/legate/lldb`` directories, respectively.

.. note::

   Pretty printers for GDB on Darwin/macOS are not currently supported.

The pretty printers support the following types:
- ``legate::detail::SmallVector``

Usage
-----

Commands to load these printers are provided in the ``.gdbinit`` and ``.lldbinit``
files at the root of the Legate repository. To load these printers automatically
when running your debugger at the project root, GDB and LLDB must be given
permission to load these init files automatically.

This can be done by adding the following to ``~/.gdbinit``:

.. code-block:: sh

   set auto-load safe-path /path/to/legate/dir

and the following to ``~/.lldbinit``:

.. code-block:: sh

   settings set target.load-cwd-lldbinit true

When running your debugger at project root, you can validate the printers
are loaded in GDB by running ``info pretty-printer``  and seeing the legate printers
are listed under the ``legate`` category:

.. code-block:: sh

   (gdb) info pretty-printer
   global pretty-printers:
     ...
     legate
       SmallVector
     ...

Similarly, you can validate the printers are loaded in LLDB by running ``type summary list``
and ``type synthetic list`` and seeing the Legate types are listed under the default category:

.. code-block:: sh

   (lldb) type synthetic list
   -----------------------
   Category: default
   -----------------------
   legate::detail::SmallVector<.*>:  Python class legate_formatters.SmallVectorChildrenProvider

To confirm the pretty printers are working, you can create a Legate object and print it:

.. code-block:: sh

   (gdb) p small_vec_var
   $1 = legate::detail::SmallVector<unsigned long, 6> of size=3, mode=small = {0, 1, 2}

   (lldb) p small_vec_var
   (legate::detail::SmallVector<unsigned long>) size=2 mode=small {
     [0] = 0
     [1] = 1
     [2] = 2
   }
