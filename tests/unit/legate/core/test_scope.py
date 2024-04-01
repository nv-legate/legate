# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import pytest

from legate.core import ExceptionMode, Machine, Scope

MAGIC_PRIORITY1 = 42
MAGIC_PRIORITY2 = 43

MAGIC_PROVENANCE1 = "42"
MAGIC_PROVENANCE2 = "43"

MODE1 = ExceptionMode.DEFERRED
MODE2 = ExceptionMode.IGNORED


def slice_if_not_singleton(machine: Machine) -> Machine:
    return machine.slice(slice(-1)) if machine.count() > 1 else machine


class TestScope:
    def test_basic_priority(self) -> None:
        old_priority = Scope.priority()
        with Scope(priority=MAGIC_PRIORITY1):
            assert Scope.priority() == MAGIC_PRIORITY1
        assert Scope.priority() == old_priority

    def test_basic_exception_mode(self) -> None:
        old_exception_mode = Scope.exception_mode()
        with Scope(exception_mode=MODE1):
            assert Scope.exception_mode() == MODE1
        assert Scope.exception_mode() == old_exception_mode

    def test_basic_provenance(self) -> None:
        old_provenance = Scope.provenance()
        with Scope(provenance=MAGIC_PROVENANCE1):
            assert Scope.provenance() == MAGIC_PROVENANCE1
        assert Scope.provenance() == old_provenance

    def test_basic_machine(self) -> None:
        old_machine = Scope.machine()
        sliced = slice_if_not_singleton(old_machine)
        with Scope(machine=sliced):
            assert Scope.machine() == sliced
        assert Scope.machine() == old_machine

    def test_basic_multiple(self) -> None:
        old_priority = Scope.priority()
        old_exception_mode = Scope.exception_mode()
        old_provenance = Scope.provenance()
        old_machine = Scope.machine()
        sliced = slice_if_not_singleton(old_machine)

        with Scope(
            priority=MAGIC_PRIORITY1,
            exception_mode=MODE1,
            provenance=MAGIC_PROVENANCE1,
            machine=sliced,
        ):
            assert Scope.priority() == MAGIC_PRIORITY1
            assert Scope.exception_mode() == MODE1
            assert Scope.provenance() == MAGIC_PROVENANCE1
            assert Scope.machine() == sliced
        assert Scope.priority() == old_priority
        assert Scope.exception_mode() == old_exception_mode
        assert Scope.provenance() == old_provenance
        assert Scope.machine() == old_machine

    def test_nested_priority(self) -> None:
        old_priority = Scope.priority()
        with Scope(priority=MAGIC_PRIORITY1):
            assert Scope.priority() == MAGIC_PRIORITY1
            with Scope(priority=MAGIC_PRIORITY2):
                assert Scope.priority() == MAGIC_PRIORITY2
            assert Scope.priority() == MAGIC_PRIORITY1
        assert Scope.priority() == old_priority

    def test_nested_exception_mode(self) -> None:
        old_exception_mode = Scope.exception_mode()
        with Scope(exception_mode=MODE1):
            assert Scope.exception_mode() == MODE1
            with Scope(exception_mode=MODE2):
                assert Scope.exception_mode() == MODE2
            assert Scope.exception_mode() == MODE1
        assert Scope.exception_mode() == old_exception_mode

    def test_nested_provenance(self) -> None:
        old_provenance = Scope.provenance()
        with Scope(provenance=MAGIC_PROVENANCE1):
            assert Scope.provenance() == MAGIC_PROVENANCE1
            with Scope(provenance=MAGIC_PROVENANCE2):
                assert Scope.provenance() == MAGIC_PROVENANCE2
            assert Scope.provenance() == MAGIC_PROVENANCE1
        assert Scope.provenance() == old_provenance

    def test_nested_machine(self) -> None:
        old_machine = Scope.machine()
        sliced1 = slice_if_not_singleton(old_machine)
        with Scope(machine=sliced1):
            assert Scope.machine() == sliced1
            sliced2 = slice_if_not_singleton(sliced1)
            with Scope(machine=sliced2):
                assert Scope.machine() == sliced2
            assert Scope.machine() == sliced1
        assert Scope.machine() == old_machine

    def test_nested_multiple(self) -> None:
        old_priority = Scope.priority()
        old_exception_mode = Scope.exception_mode()
        old_provenance = Scope.provenance()
        old_machine = Scope.machine()
        sliced1 = slice_if_not_singleton(old_machine)

        with Scope(
            priority=MAGIC_PRIORITY1,
            exception_mode=MODE1,
            provenance=MAGIC_PROVENANCE1,
            machine=sliced1,
        ):
            assert Scope.priority() == MAGIC_PRIORITY1
            assert Scope.exception_mode() == MODE1
            assert Scope.provenance() == MAGIC_PROVENANCE1
            assert Scope.machine() == sliced1

            sliced2 = slice_if_not_singleton(sliced1)
            with Scope(
                priority=MAGIC_PRIORITY2,
                exception_mode=MODE2,
                provenance=MAGIC_PROVENANCE2,
                machine=sliced2,
            ):
                assert Scope.priority() == MAGIC_PRIORITY2
                assert Scope.exception_mode() == MODE2
                assert Scope.provenance() == MAGIC_PROVENANCE2
                assert Scope.machine() == sliced2

            assert Scope.priority() == MAGIC_PRIORITY1
            assert Scope.exception_mode() == MODE1
            assert Scope.provenance() == MAGIC_PROVENANCE1
            assert Scope.machine() == sliced1

        assert Scope.priority() == old_priority
        assert Scope.exception_mode() == old_exception_mode
        assert Scope.provenance() == old_provenance
        assert Scope.machine() == old_machine

    def test_nested_with_empty_scope(self) -> None:
        old_priority = Scope.priority()
        old_exception_mode = Scope.exception_mode()
        old_provenance = Scope.provenance()
        old_machine = Scope.machine()
        sliced1 = slice_if_not_singleton(old_machine)

        with Scope(
            priority=MAGIC_PRIORITY1,
            exception_mode=MODE1,
            provenance=MAGIC_PROVENANCE1,
            machine=sliced1,
        ):
            assert Scope.priority() == MAGIC_PRIORITY1
            assert Scope.exception_mode() == MODE1
            assert Scope.provenance() == MAGIC_PROVENANCE1
            assert Scope.machine() == sliced1

            sliced2 = slice_if_not_singleton(sliced1)
            with Scope():
                with Scope(
                    priority=MAGIC_PRIORITY2,
                    exception_mode=MODE2,
                    provenance=MAGIC_PROVENANCE2,
                    machine=sliced2,
                ):
                    assert Scope.priority() == MAGIC_PRIORITY2
                    assert Scope.exception_mode() == MODE2
                    assert Scope.provenance() == MAGIC_PROVENANCE2
                    assert Scope.machine() == sliced2

            assert Scope.priority() == MAGIC_PRIORITY1
            assert Scope.exception_mode() == MODE1
            assert Scope.provenance() == MAGIC_PROVENANCE1
            assert Scope.machine() == sliced1

        assert Scope.priority() == old_priority
        assert Scope.exception_mode() == old_exception_mode
        assert Scope.provenance() == old_provenance
        assert Scope.machine() == old_machine


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
