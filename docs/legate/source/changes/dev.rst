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

C++
---

.. rubric:: General

.. rubric:: Data

.. rubric:: Mapping

- Remove ``legate::mapping::Mapper::task_target()``.

.. rubric:: Partitioning

.. rubric:: Tasks

.. rubric:: Types

.. rubric:: Runtime

- Deprecate ``legate::start(argc, argv)``. Users should use the argument-less version
  ``legate::start()`` instead. The values of ``argc`` and ``argv`` were always ignored, so
  this change has no runtime effect.

.. rubric:: Utilities

- Remove ``legate::comm::coll::collInit()`` and ``legate::comm::coll::collFinalize()``.

.. rubric:: I/O

- Move the HDF5 interface out from the experimental
  namespace. ``legate::experimental::io::hdf5`` is now ``legate::io::hdf5``.


Python
------

.. rubric:: General

- Add environment variable ``LEGATE_LIMIT_STDOUT``. If enabled, restricts `stdout` output
  to only the first rank (default is output for all ranks).

.. rubric:: Data

.. rubric:: Mapping

.. rubric:: Partitioning

.. rubric:: Tasks

.. rubric:: Types

.. rubric:: Runtime

.. rubric:: Utilities

.. rubric:: I/O

- Move the HDF5 interface out from the experimental
  namespace. ``legate.core.experimental.io.hdf5`` is now ``legate.io.hdf5``.
