# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..utilities.typedefs cimport _LocalTaskID
from ..utilities.utils cimport register_finalizer

from .variant_options cimport VariantOptions

cdef extern from * nogil:
    r"""
    namespace {

    const legate::VariantOptions *get_variant_options(
      const legate::TaskConfig &config)
    {
      auto &&opts = config.variant_options();

      if (opts.has_value()) {
        return &opts->get();
      }
      return nullptr;
    }

    } // namespace
    """
    # Need this helper because Cython does not understand reference_wrapper
    const _VariantOptions *get_variant_options(
        const _TaskConfig &config
    ) except+

cdef void _destroy_handle(void *handle_ptr):
    cdef _TaskConfig *handle = <_TaskConfig *>handle_ptr

    with nogil:
        handle[0] = _TaskConfig()

cdef class TaskConfig:
    def __init__(
        self,
        task_id: _LocalTaskID,
        *,
        options: VariantOptions | None = None
    ) -> None:
        r"""
        Construct a ``TaskConfig``.

        Parameters
        ----------
        task_id : LocalTaskID
            The local task ID for the task.
        options : VariantOptions, optional
            The task-wide default variant options, if any.
        """
        self._handle = _TaskConfig(task_id)

        if options is not None:
            self.variant_options = options

        register_finalizer(self, _destroy_handle, &self._handle)

    @property
    def task_id(self) -> _LocalTaskID:
        r"""
        Get the task ID.

        :returns: The task ID
        :rtype: LocalTasKID
        """
        cdef _LocalTaskID id

        with nogil:
            id = self._handle.task_id()

        return id

    @property
    def variant_options(self) -> VariantOptions | None:
        r"""
        Get the current variant options if they have been set.

        The returned object is a *copy* of the true variant options due to the
        fact that Python does not expose "const" views of objects. Therefore,
        any mutations performed on the returned ``VariantOptions`` will not be
        reflected in the ``TaskConfig``. To update the value, the user must
        re-set the new value:

        ::

            options = config.variant_options
            # Set some new value etc...
            options.may_throw_exception = True
            # Must re-set the value after modification in order to modify the
            # task config
            config.variant_options = options


        :returns: The variant options or None if no options have been set.
        :rtype: VariantOptions | None
        """
        cdef const _VariantOptions *opts

        with nogil:
            opts = get_variant_options(self._handle)

        if opts:
            return VariantOptions.from_handle(opts[0])
        return None

    @variant_options.setter
    def variant_options(self, options: VariantOptions) -> None:
        """
        :param options: The new variant options to set
        :type options: VariantOptions
        :returns: None
        :rtype: None
        """
        cdef VariantOptions vopts = options

        with nogil:
            self._handle.with_variant_options(vopts._handle)
