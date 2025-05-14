# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from legate.core import ExceptionMode, Machine, ParallelPolicy, Scope

MAGIC_PRIORITY1 = 42
MAGIC_PRIORITY2 = 43

MAGIC_PROVENANCE1 = "42"
MAGIC_PROVENANCE2 = "43"

MODE1 = ExceptionMode.DEFERRED
MODE2 = ExceptionMode.IGNORED

OVERDECOMPOSE_FACTOR_A = 42
OVERDECOMPOSE_FACTOR_B = 43


def par_policy_a() -> ParallelPolicy:
    return ParallelPolicy(
        streaming=True, overdecompose_factor=OVERDECOMPOSE_FACTOR_A
    )


def par_policy_b() -> ParallelPolicy:
    return ParallelPolicy(
        streaming=False, overdecompose_factor=OVERDECOMPOSE_FACTOR_B
    )


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

    def test_basic_parallel_policy(self) -> None:
        old_parallel_policy = Scope.parallel_policy()
        with Scope(parallel_policy=par_policy_a()):
            assert Scope.parallel_policy() == par_policy_a()
        assert Scope.parallel_policy() == old_parallel_policy

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

    def test_nested_parallel_policy(self) -> None:
        old_parallel_policy = Scope.parallel_policy()
        with Scope(parallel_policy=par_policy_a()):
            assert Scope.parallel_policy() == par_policy_a()
            with Scope(parallel_policy=par_policy_b()):
                assert Scope.parallel_policy() == par_policy_b()
            assert Scope.parallel_policy() == par_policy_a()
        assert Scope.parallel_policy() == old_parallel_policy

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
            with (
                Scope(),
                Scope(
                    priority=MAGIC_PRIORITY2,
                    exception_mode=MODE2,
                    provenance=MAGIC_PROVENANCE2,
                    machine=sliced2,
                ),
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
