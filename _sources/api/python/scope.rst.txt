.. currentmodule:: legate.core

Execution Scope
===============

``Scope`` allows users to specify restrictions on the runtime's behavior within a given
block of code. For example, the user may specify that a particular set of tasks may only
execute on GPUs, or that a particular set of tasks may only execute on CPUs 1 through 9.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Scope
