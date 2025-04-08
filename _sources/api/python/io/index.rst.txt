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
   kerchunk_read

Examples
--------

.. toctree::
   :maxdepth: 1

   hdf5_gds

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
