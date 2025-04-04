.. currentmodule:: legate.core

Task Context
============

The ``TaskContext`` class contains the inputs, outputs, scalars, reductions, and other
task information during a task's execution. It is normally hidden from users using pure
:ref:`Python tasks <label_tasks>`, but can be explicitly requested via a task
argument. See :ref:`label_explicit_task_context_argument` for further discussion.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   TaskContext
