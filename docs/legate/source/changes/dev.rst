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

- Add an environment variable ``LEGATE_MAX_EXCEPTION_SIZE`` that determines the maximum number of
  bytes for an exception that can be raised by a task

C++
---

.. rubric:: General

.. rubric:: Data

.. rubric:: Mapping

.. rubric:: Partitioning

.. rubric:: Tasks

- Remove ``legate::VariantOptions::return_size``.

.. rubric:: Types

.. rubric:: Runtime

- Deprecate ``legate::start(argc, argv)``. Users should use the argument-less version
  ``legate::start()`` instead. The values of ``argc`` and ``argv`` were always ignored, so
  this change has no runtime effect.
- Add exception types ``legate::ConfigurationError`` and
  ``legate::AutoConfigurationError`` to signal Legate configuration failures.
- Add ``legate::Runtime::start_profiling_range`` and
  ``legate::Runtime::stop_profiling_range`` to create Legion profile ranges.

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
