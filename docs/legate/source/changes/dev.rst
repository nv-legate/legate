Changes: Latest Development Version
===================================

..
   STYLE:
   * Capitalize sentences.
   * Use the imperative tense: Add, Improve, Change, etc.
   * Use a period (.) at the end of entries.
   * Be concise yet informative.
   * If possible, provide an executive summary of the new feature, but do not
     just repeat its doc string. However, if the feature requires changes from
     the user, then describe those changes in detail, and provide examples of
     the changes required.


.. rubric:: General

- Add an environment variable ``LEGATE_MAX_EXCEPTION_SIZE`` that determines the maximum
  number of bytes for an exception that can be raised by a task.
- Improve specification of logging levels. These may now be specified in a more
  human-readable manner. Instead of ``--logging some_logger=2``, the user may now pass
  ``--logging some_logger=info``. To see supported values for this feature, see the
  ``--help`` output of the legate driver, or by running with ``LEGATE_CONFIG=--help``. To
  ease adoption, the old numeric logging values continue to be supported.

C++
---

.. rubric:: General

.. rubric:: Data

.. rubric:: Mapping

.. rubric:: Partitioning

.. rubric:: Tasks

- Remove ``legate::VariantOptions::return_size``.
- Add ``legate::TaskInfo``. This class has technically always existed, but was
  undocumented.
- Add ``legate::VariantInfo``. This class has technically always existed, but was
  undocumented.

.. rubric:: Types

.. rubric:: Runtime

- Add ``legate::Runtime::start_profiling_range`` and
  ``legate::Runtime::stop_profiling_range`` to create Legion profile ranges.
- Change ``legate::Library::register_task()``. It now takes a ``const legate::TaskInfo &``
  instead of a ``std::unique_ptr<legate::TaskInfo>``
- Change ``legate::Library::find_task()``. It now returns a ``legate::TaskInfo`` instead
  of a ``const legate::TaskInfo *``.

.. rubric:: Utilities

.. rubric:: I/O


Python
------

.. rubric:: General

.. rubric:: Data

.. rubric:: Mapping

.. rubric:: Partitioning

.. rubric:: Tasks

- Add ``legate.core.ProfileRange`` context manager to activate the API for
  generating sub-boxes on the profiler.

.. rubric:: Types

.. rubric:: Runtime

.. rubric:: Utilities

.. rubric:: I/O
