Changes: Latest Development Version
===================================

..
   STYLE:
   * Capitalize sentences.
   * Use the imperative tense: Add, Improve, Change, etc.
   * Use a period (.) at the end of entries.
   * Be concise yet informative.
   * If possible, provide an executive summary of the new feature, but do not just repeat
     its doc string. However, if the feature requires changes from the user, then describe
     those changes in detail, and provide examples of the changes required.


.. rubric:: General

- Drop support for Maxwell GPU architecture. Legate now requires at least Pascal
  (``sm_60``).
- Change the default number of maximum dimensions for Legate arrays from 4 to 6.
- Remove the ``--eager-alloc-percentage`` flag, as the one pool allocation scheme
  makes it obsolete.

C++
---

.. rubric:: General

.. rubric:: Data

- Add ``legate::LogicalStore::offload_to()`` and
  ``legate::LogicalArray::offload_to()`` to allow offloading a store, or
  an array, to a particular memory kind, such that any copies in other
  memories are discarded.

- Add ``legate::LogicalStore::reinterpret_as()`` to reinterpret the underlying buffer of a
  ``LogicalStore`` as another data-type.

.. rubric:: Mapping

- Add ``legate::mapping::Mapper::allocation_pool_size()``. Legate mappers are
  now required to implement the new mapping callback returning sizes of
  allocation pools used by task variants that are registered with
  ``has_allocations`` being ``true``.

- Add ``legate::mapping::Operation::num_scalars()``.

- Add ``legate::mapping::Operation::is_single_task()``.

- Add ``legate::mapping::Operation::get_launch_domain()``.

- Remove ``legate::mapping::Mapper::task_target()``.

.. rubric:: Partitioning

.. rubric:: Tasks

- Remove ``legate::VariantOptions::leaf``, ``legate::VariantOptions::inner``,
  and ``legate::VariantOptions::idempotent``, as they don't actually do
  anything in Legate.

- Add ``legate::VariantOptions::has_allocations``, which indicates that the
  task variant is allowed to create temporary or output buffers during
  execution.

.. rubric:: Types

- Add support for enums to ``legate::type_code_of``. ``legate::type_code_of`` will now
  unwrap the type to its underlying type if it is an enum. As a result, many objects (like
  ``legate::Scalar``) now natively support enums. For example, what previously required:

  .. code-block:: cpp

     enum class MyEnum { FOO };

     auto scal = legate::Scalar{
       static_cast<std::underlying_type_t<MyEnum>>(MyEnum::FOO)
     };

     auto value = scal.value<std::underlying_type_t<MyEnum>>();

  May now be done directly:

  .. code-block:: cpp

     enum class MyEnum { FOO };

     auto scal = legate::Scalar{MyEnum::FOO};

     auto value = scal.value<MyEnum>();


.. rubric:: Runtime

- Deprecate ``legate::start(argc, argv)``. Users should use the argument-less version
  ``legate::start()`` instead. The values of ``argc`` and ``argv`` were always ignored, so
  this change has no runtime effect.
- Add exception types ``legate::ConfigurationError`` and
  ``legate::AutoConfigurationError`` to signal Legate configuration failures.

.. rubric:: Utilities

- Remove ``legate::comm::coll::collInit()`` and ``legate::comm::coll::collFinalize()``.
- Remove ``legate::VariantCode::NONE``, and ``LEGATE_VARIANT_NONE``. They served no
  purpose and were not used.

.. rubric:: I/O

- Move the HDF5 interface out from the experimental
  namespace. ``legate::experimental::io::hdf5`` is now ``legate::io::hdf5``.


Python
------

.. rubric:: General

- Add environment variable ``LEGATE_LIMIT_STDOUT``. If enabled, restricts `stdout` output
  to only the first rank (default is output for all ranks).

.. rubric:: Data

- Add ``legate.core.LogicalStore.offload_to()`` and
  ``legate.core.LogicalArray.offload_to()`` to allow offloading a store, or an
  array, to a particular memory kind, such that any copies in other memories are
  discarded.

.. rubric:: Mapping

.. rubric:: Partitioning

.. rubric:: Tasks

.. rubric:: Types

.. rubric:: Runtime

.. rubric:: Utilities

- Remove ``legate.core.VariantCode.NONE``. It served no purpose and was never used.

.. rubric:: I/O

- Move the HDF5 interface out from the experimental
  namespace. ``legate.core.experimental.io.hdf5`` is now ``legate.io.hdf5``.
