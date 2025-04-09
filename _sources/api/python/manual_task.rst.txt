.. currentmodule:: legate.core

Manual Tasks
============

Where ``AutoTask`` s are parallelized automatically by the runtime, ``ManualTask`` is
parallelized explicitly by the user. Usually it suffices to allow the runtime to
parallelize a task, but sometimes a more fine-grained approach is needed. In this case, a
user may construct a ``ManualTask`` and describe exactly the manner in which arguments are
partitioned.


.. autosummary::
   :toctree: generated/
   :template: class.rst

   ManualTask
