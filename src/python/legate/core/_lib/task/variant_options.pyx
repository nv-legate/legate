# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.string cimport string as std_string
from libcpp.utility cimport move as std_move

from ..._ext.cython_libcpp.string_view cimport (
    std_string_view,
    str_from_string_view
)

from typing import Sequence

cdef extern from *:
    """
    #include <deque>
    #include <cstddef>
    #include <vector>
    #include <string>

    namespace legate::cython_detail {

    std::size_t get_comm_array_size(const legate::VariantOptions& options)
    {
      return options.communicators.value().size();
    }

    std::string_view get_comm_at_idx(
      const legate::VariantOptions& options, std::size_t idx)
    {
      return options.communicators.value()[idx];
    }

    // VariantOptions does not take ownership of the communicators, because it
    // is meant to be used with initializer-list + static storage duration
    // strings. Since that's not possible in Python, we need to define this
    // global variable that holds all of the strings set until the end of the
    // program so that they don't ever go out of scope.
    std::deque<std::vector<std::string>> GLOBAL_COMMS_BUFFER;

    void set_new_comms(
      std::vector<std::string> comms,
      legate::VariantOptions* options)
    {
      auto& gcomms = GLOBAL_COMMS_BUFFER.emplace_back(std::move(comms));

      options->with_communicators({}, gcomms.begin(), gcomms.end());
    }

    } // namespace legate::cython_detail
    """
    size_t _get_comm_array_size "legate::cython_detail::get_comm_array_size" (
        const _VariantOptions& options
    ) except+
    std_string_view _get_comm_at_idx "legate::cython_detail::get_comm_at_idx" (
        const _VariantOptions& options, size_t idx
    ) except+
    void _set_new_comms "legate::cython_detail::set_new_comms" (
        std_vector[std_string] comms,
        _VariantOptions *options
    ) except+

cdef class VariantOptions:
    @staticmethod
    cdef VariantOptions from_handle(const _VariantOptions& handle):
        cdef VariantOptions ret = VariantOptions.__new__(VariantOptions)

        ret._handle = handle
        return ret

    def __init__(
        self,
        *,
        concurrent = None,
        has_allocations = None,
        elide_device_ctx_sync = None,
        has_side_effect = None,
        may_throw_exception = None,
        communicators: Sequence[str] | None = None
    ) -> None:
        self._handle = _VariantOptions()

        if concurrent is not None:
            self.concurrent = concurrent
        if has_allocations is not None:
            self.has_allocations = has_allocations
        if elide_device_ctx_sync is not None:
            self.elide_device_ctx_sync = elide_device_ctx_sync
        if may_throw_exception is not None:
            self.may_throw_exception = may_throw_exception
        if communicators is not None:
            self.communicators = communicators

    def __eq__(self, object rhs) -> bool:
        if not isinstance(rhs, VariantOptions):
            return NotImplemented

        cdef VariantOptions rhs_v = rhs

        return self._handle == rhs_v._handle

    @property
    def concurrent(self) -> bool:
        """
        :returns: The current value of the concurrent flag.
        :rtype: bool
        """
        return self._handle.concurrent

    @concurrent.setter
    def concurrent(self, concurrent: bool) -> None:
        """
        :param concurrent: The value to set for the concurrent flag.
        :type concurrent: bool
        :returns: None
        :rtype: None
        """
        self._handle.with_concurrent(concurrent)

    @property
    def has_allocations(self) -> bool:
        """
        :returns: The current value of the has_allocations flag.
        :rtype: bool
        """
        return self._handle.has_allocations

    @has_allocations.setter
    def has_allocations(self, has_allocations: bool) -> None:
        """
        :param has_allocations: The value to set for the has_allocations flag.
        :type has_allocations: bool
        :returns: None
        :rtype: None
        """
        self._handle.with_has_allocations(has_allocations)

    @property
    def elide_device_ctx_sync(self) -> bool:
        """
        :returns: The current value of the elide_device_ctx_sync flag.
        :rtype: bool
        """
        return self._handle.elide_device_ctx_sync

    @elide_device_ctx_sync.setter
    def elide_device_ctx_sync(self, elide_sync: bool) -> None:
        """
        :param elide_sync: The value to set for the elide_device_ctx_sync flag.
        :type elide_sync: bool
        :returns: None
        :rtype: None
        """
        self._handle.with_elide_device_ctx_sync(elide_sync)

    @property
    def has_side_effect(self) -> bool:
        """
        :returns: The current value of the has_side_effect flag.
        :rtype: bool
        """
        return self._handle.has_side_effect

    @has_side_effect.setter
    def has_side_effect(self, side_effect: bool) -> None:
        """
        :param side_effect: The value to set for the has_side_effect flag.
        :type side_effect: bool
        :returns: None
        :rtype: None
        """
        self._handle.with_has_side_effect(side_effect)

    @property
    def may_throw_exception(self) -> bool:
        """
        :returns: The current value of the may_throw_exception flag.
        :rtype: bool
        """
        return self._handle.may_throw_exception

    @may_throw_exception.setter
    def may_throw_exception(self, may_throw: bool) -> None:
        """
        :param may_throw: The value to set for the may_throw_exception flag.
        :type may_throw: bool
        :returns: None
        :rtype: None
        """
        self._handle.with_may_throw_exception(may_throw)

    @property
    def communicators(self) -> tuple[str, ...]:
        """
        :returns: A list of communicators (as strings).
        :rtype: tuple[str, ...]
        """
        cdef list tmp = []
        cdef std_string_view comm
        cdef size_t i, size

        if self._handle.communicators.has_value():
            # Have to do it this way (instead of for comm in
            # communicators.value()) because Cython doesn't understand
            # std::array.
            size = _get_comm_array_size(self._handle)
            for i in range(size):
                comm = _get_comm_at_idx(self._handle, i)
                if not comm.empty():
                    tmp.append(str_from_string_view(comm))

        return tuple(tmp)

    @communicators.setter
    def communicators(self, comms: Sequence[str]) -> None:
        """
        :param comms: A list of communicators to set.
        :type comms: Sequence[str]
        :returns: None
        :rtype: None
        """
        cdef list[bytes] py_comms = [c.encode() for c in comms]
        cdef std_vector[std_string] cpp_comms

        cpp_comms.reserve(len(py_comms))
        cpp_comms = py_comms

        # Have to use a special function like this because Cython doesn't
        # understand std::initializer_list
        _set_new_comms(std_move(cpp_comms), &self._handle)
