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

.. rubric:: Data

.. rubric:: Mapping

.. rubric:: Partitioning

.. rubric:: Tasks

- Add ``VariantOptions::elide_device_ctx_sync`` and
  ``VariantOptions::with_elide_device_ctx_sync()`` to allow specifying that a particular
  task variant need not perform device context synchronization after task completion.
- Add ``TaskContext::get_task_stream()`` to retrieve the current tasks' active CUDA stream.
- Deprecate ``TaskInfo::add_variant()``. Users are encouraged to wrap their tasks in
  ``LegateTask`` and call ``LegateTask::register_variants()`` instead to handle the
  various task variants registration steps.
- Deprecate ``TaskRegistrar::record_task(std::int64_t std::unique_ptr<TaskInfo>)``. Users
  should use ``LegateTask::Registrar`` combined with ``TaskRegistrar::record_all_task()``
  instead.
- Change ``TaskRegistrar::register_all_tasks()`` to take a mutable reference to the
  ``library`` argument instead of by value.

.. rubric:: Types

.. rubric:: Runtime

- Add optional ``default_options`` argument to ``Runtime::create_library()`` to specify
  the library-default task variant options.
- Add optional ``default_options`` argument to ``Runtime::find_or_create_library()`` to
  specify the library-default task variant options.
- Add ``Library::get_default_variant_options()`` to retrieve the library-default task
  variant options.

.. rubric:: Utilities

- Deprecate ``legate::cuda::StreamPool``. Users should use
  ``TaskContext::get_task_stream()`` instead within tasks.
- Deprecate ``legate::cuda::StreamView``. Users should implement their own version of this
  class.
