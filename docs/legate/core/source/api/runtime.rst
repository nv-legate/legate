.. _label_runtime:

.. currentmodule:: legate.core

Legate Runtime
==============

The Legate runtime provides APIs for creating stores and issuing tasks and
other kinds of operations.

.. autosummary::
   :toctree: generated/

   Runtime.create_store
   Runtime.create_manual_task
   Runtime.create_auto_task
   Runtime.issue_fill
   Runtime.issue_execution_fence
   Runtime.tree_reduce


Annotation
----------

An ``Annotation`` is a context manager to set library specific annotations that
are to be attached to operations issued within a scope. A typical usage of
``Annotation`` would look like this:

::

  with Annotation(lib_context, { "key1" : "value1", "key2" : "value2", ... }:
    ...

Then each operation in the scope is annotated with the key-value pairs,
which are later rendered in execution profiles.

.. autosummary::
   :toctree: generated/

   Annotation.__init__
