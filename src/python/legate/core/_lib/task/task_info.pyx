# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.utility cimport move as std_move

from ..._ext.cython_libcpp.string_view cimport (
    str_from_string_view,
    std_string_view
)
from ..._ext.task.python_task cimport (
    register_variants,
    finalize_variant_registration
)

from ..runtime.library cimport Library
from ..runtime.runtime cimport get_legate_runtime
from ..utilities.typedefs cimport VariantCode, _GlobalTaskID
from ..utilities.unconstructable cimport Unconstructable
from .task_config cimport TaskConfig


cdef class TaskInfo(Unconstructable):
    @staticmethod
    cdef TaskInfo from_handle(_TaskInfo handle, _LocalTaskID local_task_id):
        cdef TaskInfo result = TaskInfo.__new__(TaskInfo)

        result._handle = std_move(handle)
        result._local_id = local_task_id
        result._registered_variants = {}
        return result

    cdef void validate_registered_py_variants(self):
        if not self._registered_variants:
            raise RuntimeError(
                f"Task (local id: {self.get_local_id()}) has no variants!"
            )

    cdef void register_global_variant_callbacks(
        self,
        _GlobalTaskID global_task_id
    ):
        register_variants(global_task_id, self._registered_variants)
        self._registered_variants = {}

    cdef _LocalTaskID get_local_id(self):
        return self._local_id

    def __repr__(self) -> str:
        r"""
        Return a human-readable string representation of the ``TaskInfo``.

        Returns
        -------
        str
            The string representation.
        """
        # must regular import here to get the python enum version
        from ..utilities.typedefs import VariantCode as py_VariantCode

        cdef list[str] descr = [
            vid.name
            for vid in py_VariantCode
            if self.has_variant(vid)
        ]
        cdef str variants = ", ".join(descr) if descr else "None"
        return f"TaskInfo(name: {self.name}, variants: {variants})"

    # The exact same thing as from_variants() (in fact, that function calls
    # this one), except that it also accepts a TaskConfig argument. The
    # reason we have two versions of the function is because TaskConfig does
    # not need to be exposed to the user.
    @staticmethod
    cdef TaskInfo from_variants_config(
        TaskConfig config,
        Library library,
        str name,
        list[tuple[VariantCode, object]] variants,
    ):
        cdef _LocalTaskID task_id = config.task_id

        if not variants:
            m = (
                "Cannot construct task info "
                f"(local id: {task_id}, name: {name})."
                " Variants must not be empty."
            )
            raise ValueError(m)

        cdef TaskInfo task_info

        task_info = TaskInfo.from_handle(
            _TaskInfo(name.encode()), task_id
        )

        cdef VariantCode variant_kind
        for variant_kind, variant_fn in variants:
            task_info.add_variant_config(
                config=config,
                library=library,
                variant_kind=variant_kind,
                fn=variant_fn
            )
        return task_info

    @classmethod
    def from_variants(
        cls,
        _LocalTaskID local_task_id,
        str name,
        list[tuple[VariantCode, object]] variants
    ) -> TaskInfo:
        r"""
        Construct a ``TaskInfo`` from a list of variants.

        Parameters
        ----------
        local_task_id : LocalTaskID
            The local task ID of the task.
        name : str
            The name of the task.
        variants : list[tuple[VariantCode, VariantFunction]]
            The variants to register.

        Returns
        -------
        TaskInfo
            The created ``TaskInfo`` object.
        """
        return TaskInfo.from_variants_config(
            config=TaskConfig(local_task_id),
            library=get_legate_runtime().core_library,
            name=name,
            variants=variants
        )

    @property
    def name(self) -> str:
        r"""
        Get the name of the task.

        :returns: The task name.
        :rtype: str

        :raises RuntimeError: If the task info object is in an invalid state.
        """
        cdef std_string_view sv

        with nogil:
            sv = self._handle.name()
        return str_from_string_view(sv)

    cpdef bool has_variant(self, VariantCode variant_id):
        r"""
        Get whether a `TaskInfo` object has a particular variant.

        Parameters
        ----------
        variant_id : VariantCode
            The variant kind to query.

        Returns
        -------
        bool
            ``True`` if the variant exists, ``False`` otherwise.

        Raises
        ------
        RuntimeError
            If the task info object is in an invalid state.
        """
        cdef bool ret

        with nogil:
            ret = self._handle.find_variant(variant_id).has_value()
        return ret

    cdef void add_variant_config(
        self,
        TaskConfig config,
        Library library,
        VariantCode variant_kind,
        object fn
    ):
        if not callable(fn):
            raise TypeError(
                f"Variant function ({fn}) for variant kind {variant_kind} is "
                "not callable"
            )

        if variant_kind in self._registered_variants:
            raise RuntimeError(
                "Already added callback "
                f"({self._registered_variants[variant_kind]}) "
                f"for {variant_kind} variant "
                f"(local id: {self.get_local_id()})"
            )

        finalize_variant_registration(
            task_info=self, config=config, library=library, code=variant_kind
        )
        # do this last to preserve strong exception guarantee
        self._registered_variants[variant_kind] = fn

    cpdef void add_variant(self, VariantCode variant_kind, object fn):
        r"""
        Register a variant to a `TaskInfo`.

        Parameters
        ----------
        variant_kind : VariantCode
            The variant kind to add.
        fn : VariantFunction
            The variant to add.

        Raises
        ------
        RuntimeError
            If the task info object is in an invalid state.
        TypeError
            If ``fn`` is not callable.
        ValueError
            If ``variant_kind`` is an unknown variant kind.
        RuntimeError
            If the task info object has already registered a variant for
            ``variant_kind``.
        """
        self.add_variant_config(
            config=TaskConfig(self._local_id),
            library=get_legate_runtime().core_library,
            variant_kind=variant_kind,
            fn=fn
        )
