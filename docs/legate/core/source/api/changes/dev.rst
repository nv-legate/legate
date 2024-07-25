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

.. rubric:: Data

.. rubric:: Mapping

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

.. rubric:: Runtime

- Add optional ``default_options`` argument to ``Runtime::create_library()`` to specify
  the library-default task variant options.
- Add optional ``default_options`` argument to ``Runtime::find_or_create_library()`` to
  specify the library-default task variant options.
- Add ``Library::get_default_variant_options()`` to retrieve the library-default task
  variant options.
- Add ``Runtime::issue_mapping_fence()`` to issue a mapping fence that prevents
  all the downstream tasks from being mapped ahead of the fence.

.. rubric:: Utilities

- Deprecate ``legate::cuda::StreamPool``. Users should use
  ``TaskContext::get_task_stream()`` instead within tasks.
- Deprecate ``legate::cuda::StreamView``. Users should implement their own version of this
  class.
