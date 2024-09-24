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

- Rename the library from ``legate.core`` to ``legate``. With the renaming, the runtime does not
  automatically start with ``import legate`` anymore. Downstream libraries are now responsible to
  call ``legate.get_legate_runtime()`` before they register themselves through a C++ callback.
- Add ``LEGATE_INLINE_TASK_LAUNCH`` environment variable to request inline task
  launch. When enabled, this instructs Legate to forgo the usual Legion task calling
  convention in favor of launching the task immediately on the submitting thread. This
  feature is currently considered *experimental* as it is not always profitable when it
  should be.


C++
---

.. rubric:: General

- Remove ``legate_c.h`` and all macros defined therein.
- Remove ``legate_preamble.h``.
- Deprecate ``LEGATE_NO_VARIANT``, ``LEGATE_CPU_VARIANT``, ``LEGATE_GPU_VARIANT``, and
  ``LEGATE_OMP_VARIANT``. Users should use the corresponding ``legate::VariantCode``
  instead.
- Deprecate ``LegateVariantCode``. Users should use ``legate::VariantCode`` instead.
- Deprecate ``legate::destroy()``. Users should call ``legate::finish()`` instead.
- Add ``legate::has_finished()``.
- Add ``LocalTaskID`` and ``GlobalTaskID``, to describe local and global Task ID's. Use
  these across the public and private API whenever local or global Task ID's are
  requested. Unfortunately, this change could not be made in a backwards-compatible way,
  and users are therefore required to make the following changes. Users that previously did

  .. code-block:: c++

     struct MyTask : LegateTask<MyTask> {
       static constexpr auto TASK_ID = 0;
     };

     runtime->create_task(..., MyTask::TASK_ID);

  Must now either

  .. code-block:: c++

     struct MyTask : LegateTask<MyTask> {
       static constexpr auto TASK_ID = legate::LocalTaskID{0};

  or

  .. code-block:: c++

     runtime->create_task(..., legate::LocalTaskID{MyTask::TASK_ID});

- Add ``LocalRedopID`` and ``GlobalRedopID``, to describe local and global reduction
  operator ID's.
- Remove ``partition.h`` and ``restriction.h`` from the public API
- Add ``LEGATE_MPI_WRAPPER`` environment variable.

.. rubric:: Data

- Improve ``ScopedAllocator::ScopeAllocator()`` alignment validation. The alignment is now
  required to be a strictly positive power of 2. This was technically always the case, but
  previous versions did not check this, leading to silent failures down the line. The
  constructor will now validate the alignment and throw an exception if it is improper.
- Change passing ``nullptr`` to ``ScopedAllocator::deallocate()``. This is now a no-op
  (previously this threw a ``std::invalid_argument`` exception).
- Clarify that ``ScopedAllocator::allocate(0)`` will return a ``nullptr``. It always did
  this, but this behavior is now explicitly documented. Coupled with the change above, it
  means that all pointers returned from ``allocate()`` are safe to pass to
  ``deallocate()``.

.. rubric:: Mapping

- Add ``NodeRange::hash()``.
- Remove ``mapping::Mapper::set_machine()``.
- Add ``mapping::InstanceMappingPolicy::redundant`` that forces eager collection of freshly created
  redundant copies. Add ``mapping::InstanceMappingPolicy::with_redundant()`` and
  ``mapping::InstanceMappingPolicy::set_redundant()`` to allow mappers to set new values to the
  flag.
- Change enum values for ``mapping::TaskTarget``, ``mapping::StoreTarget``,
  ``mapping::AllocPolicy``, ``mapping::InstLayout``, and
  ``mapping::DimOrdering::Kind``. Users should only consider the names of enums (and their
  members) to be stable, and should not depend on the values themselves.

.. rubric:: Partitioning

.. rubric:: Tasks

- Add ``VariantOptions::elide_device_ctx_sync`` and
  ``VariantOptions::with_elide_device_ctx_sync()`` to allow specifying that a particular
  task variant need not perform device context synchronization after task completion.
- Add ``TaskContext::get_task_stream()`` to retrieve the current tasks' active CUDA stream.
- Deprecate ``TaskRegistrar::record_task(std::int64_t std::unique_ptr<TaskInfo>)``. Users
  should use ``LegateTask::Registrar`` combined with ``TaskRegistrar::record_all_task()``
  instead.
- Change ``TaskRegistrar::register_all_tasks()`` to take a mutable reference to the
  ``library`` argument instead of by value.
- Deprecate ``TaskInfo::has_variant()``. Users should use ``TaskInfo::find_variant()``
  directly instead.
- Change ``TaskInfo::find_variant()`` to return a ``std::optional``. If the optional has a
  value, the find succeeded and the contained value is the ``VariantInfo``. Otherwise the
  optional does not contain a value.
- Add ``TaskContext::num_scalars()`` to query the number of ``Scalar`` arguments for a
  task.
- Move the implementation detail of the CPU communicator (i.e.,
  ``legate::comm::coll::BackenedNetwork``,
  ``legate::comm::coll::LocalNetwork``, and ``legate::comm::coll::MPINetwork``)
  to the detail namespace. As a consequence, the following headers are removed
  from the public interface:``backend_network.h``, ``thread_comm.h``,
  ``local_network.h``, and ``mpi_network.h``.

.. rubric:: Types

- Removed ``ReductionOpKind::DIV`` and ``ReductionOpKind::SUB``. Partial reduction results
  are combined in an arbitrary order; since division and subtraction are neither
  commutative nor associative, it is impossible to use these reliably as reduction
  operators.
- Change the return type of ``legate::array_type`` from ``legate::Type`` to
  ``legate::FixedArrayType``.
- Change the return type of ``legate::struct_type`` from ``legate::Type`` to
  ``legate::StructType``.
- Change the return type of ``legate::list_type`` from ``legate::Type`` to
  ``legate::ListType``.
- Change the return type of ``legate::point_type`` from ``legate::Type`` to
  ``legate::FixedArrayType``. ``legate.core.types.point_type`` now returns
  ``legate.core.types.FixedArrayType`` instead of ``legate.core.types.Type``.
- Change the return type of ``legate::rect_type`` from ``legate::Type`` to ``legate::StructType``.
  ``legate.core.types.rect_type`` now returns ``legate.core.types.StructType`` instead of
  ``legate.core.types.Type``.


.. rubric:: Runtime

- Add optional ``default_options`` argument to ``Runtime::create_library()`` to specify
  the library-default task variant options.
- Add optional ``default_options`` argument to ``Runtime::find_or_create_library()`` to
  specify the library-default task variant options.
- Add ``Library::get_default_variant_options()`` to retrieve the library-default task
  variant options.
- Add ``Runtime::issue_mapping_fence()`` to issue a mapping fence that prevents
  all the downstream tasks from being mapped ahead of the fence.
- Remove ``Library::get_mapper_id()``. All libraries share the same (internal) underlying
  Legion mapper, so this method is pointless.
- Remove ``Library::register_mapper()``. The mapper should be passed when the library is
  created, and can no longer be changed after the fact.

.. rubric:: Utilities

- Deprecate ``legate::cuda::StreamPool``. Users should use
  ``TaskContext::get_task_stream()`` instead within tasks.
- Deprecate ``legate::cuda::StreamView``. Users should implement their own version of this
  class.


Python
------

.. rubric:: General

.. rubric:: Data

.. rubric:: Mapping

.. rubric:: Partitioning

.. rubric:: Tasks

- Add support for default values on scalar arguments in Python tasks.
- Add support for ``= None`` as a default argument for store-type arguments in Python
  tasks. ``None`` is currently the only allowed default value for stores. Users may use
  any of the supported "optional" type hint variants (``x | None``, ``Union[x, None]``,
  ``Optional[x]``) to inform the runtime that a store argument may be ``None``.

.. rubric:: Types

.. rubric:: Runtime

.. rubric:: Utilities
