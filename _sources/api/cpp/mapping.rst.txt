~~~~~~~
Mapping
~~~~~~~

Mapping tasks that create buffers during execution
--------------------------------------------------

From the 25.01 release, a task variant that creates temporary or output buffers
during execution requires the following two steps for correct mapping:

- The variant should be registered with a ``VariantOptions`` with the
  ``has_allocations`` field set to ``true`` (the default value is ``false``).

- The mapper should return an upper bound of the total size of allocations in
  the ``allocation_pool_size()`` call. The mapper can choose to give an
  "unbounded" allocation pool by returning ``std::nullopt``. This is always a
  sound answer to give from the mapper, but incurs performance penalty that
  mapping of any downstream tasks creating fresh allocations is blocked.

The allocation pool size is specific to each kind of memory to which the
executing processor has affinity, and the mapper is queried once for each
memory kind. The runtime does not call ``allocation_pool_size()`` for task
variants registered with ``has_allocations`` being ``false``.

.. doxygengroup:: mapping
    :members:
