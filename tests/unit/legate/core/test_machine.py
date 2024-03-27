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

from legate.core import (
    EmptyMachineError,
    Machine,
    ProcessorRange,
    ProcessorSlice,
    TaskTarget,
)


class TestTaskTargetKind:
    def test_names(self) -> None:
        assert set(k.name for k in TaskTarget) == {"GPU", "OMP", "CPU"}

    def test_values(self) -> None:
        assert list(TaskTarget) == [1, 2, 3]


class TestProcessorRange:
    def test_create_nonempty(self) -> None:
        r = ProcessorRange.create(low=1, high=3, per_node_count=1)
        assert not r.empty
        assert r.per_node_count == 1 and r.low == 1 and r.high == 3
        assert len(r) == 2

        assert r.get_node_range() == (1, 3)

    def test_create_empty(self) -> None:
        r = ProcessorRange.create(low=1, high=0, per_node_count=1)
        assert r.empty
        assert r.per_node_count == 1 and r.low == 0 and r.high == 0
        assert len(r) == 0

        r = ProcessorRange.create(low=2, high=1, per_node_count=1)
        assert r.empty
        assert r.low == 0 and r.high == 0
        assert len(r) == 0

        r = ProcessorRange.create_empty()
        assert r.empty
        assert r.low == 0 and r.high == 0
        assert len(r) == 0

        err_msg = "Illegal to get a node range of an empty processor range"
        with pytest.raises(ValueError, match=err_msg):
            r.get_node_range()

    def test_intersection_nonempty(self) -> None:
        r1 = ProcessorRange.create(low=0, high=3, per_node_count=1)
        r2 = ProcessorRange.create(low=2, high=4, per_node_count=1)
        r3 = r1 & r2
        assert r3 == ProcessorRange.create(low=2, high=3, per_node_count=1)

    def test_intersection_empty(self) -> None:
        r1 = ProcessorRange.create(low=0, high=2, per_node_count=1)
        r2 = ProcessorRange.create(low=3, high=5, per_node_count=1)
        r3 = r1 & r2
        assert r3 == ProcessorRange.create(low=1, high=0, per_node_count=1)
        assert len(r3) == 0

    def test_empty_slice_empty_range(self) -> None:
        r = ProcessorRange.create(low=3, high=1, per_node_count=1)
        assert len(r[0:0]) == 0
        assert len(r.slice(slice(0, 0))) == 0
        assert len(r[:0]) == 0
        assert len(r.slice(slice(None, 0))) == 0
        assert len(r[4:]) == 0
        assert len(r.slice(slice(4, None))) == 0

    def test_empty_slice_nonempty_range(self) -> None:
        r = ProcessorRange.create(low=2, high=5, per_node_count=1)
        assert len(r[0:0]) == 0
        assert len(r.slice(slice(0, 0))) == 0
        assert len(r[:0]) == 0
        assert len(r.slice(slice(None, 0))) == 0
        assert len(r[5:]) == 0
        assert len(r.slice(slice(5, None))) == 0

    def test_nonempty_slice_empty_range(self) -> None:
        r = ProcessorRange.create(low=3, high=1, per_node_count=1)
        assert len(r[:]) == 0
        assert len(r.slice(slice(None))) == 0
        for i in range(len(r)):
            assert len(r[:i]) == 0
            assert len(r.slice(slice(i))) == 0
            assert len(r[i:]) == 0
            assert len(r.slice(slice(i, None))) == 0

    def test_nonempty_slice_nonempty_range(self) -> None:
        r = ProcessorRange.create(low=3, high=5, per_node_count=1)
        assert len(r[:]) == len(r)
        assert len(r.slice(slice(None))) == len(r)
        for i in range(len(r)):
            assert len(r[:i]) == i
            assert len(r.slice(slice(i))) == i
            assert len(r[i:]) == len(r) - i
            assert len(r.slice(slice(i, None))) == len(r) - i

    def test_invalid_slice(self) -> None:
        r = ProcessorRange.create(low=2, high=4, per_node_count=1)
        with pytest.raises(ValueError, match="The slicing step must be 1"):
            r.slice(slice(None, None, 2))


CPU_RANGE = ProcessorRange.create(low=1, high=3, per_node_count=4)
OMP_RANGE = ProcessorRange.create(low=0, high=3, per_node_count=2)
GPU_RANGE = ProcessorRange.create(low=3, high=6, per_node_count=3)
EMPTY_RANGE = ProcessorRange.create_empty()
RANGES = [CPU_RANGE, OMP_RANGE, GPU_RANGE]
TARGETS = [TaskTarget.CPU, TaskTarget.OMP, TaskTarget.GPU]


class TestMachine:
    def test_empty_machine(self) -> None:
        m = Machine()
        assert m.preferred_target == TaskTarget.CPU
        assert len(m) == 0
        assert len(m.get_processor_range(TaskTarget.CPU)) == 0
        err_msg = "Illegal to get a node range of an empty processor range"
        with pytest.raises(ValueError, match=err_msg):
            m.get_node_range()
        with pytest.raises(ValueError, match=err_msg):
            m.get_node_range(TaskTarget.GPU)
        with pytest.raises(ValueError, match=err_msg):
            m.get_node_range(TaskTarget.OMP)
        with pytest.raises(ValueError, match=err_msg):
            m.get_node_range(TaskTarget.CPU)

    def test_eq(self) -> None:
        m1 = Machine({TaskTarget.CPU: CPU_RANGE, TaskTarget.OMP: OMP_RANGE})
        m2 = Machine({TaskTarget.CPU: CPU_RANGE, TaskTarget.OMP: OMP_RANGE})
        assert m1 == m2

        m3 = Machine()
        assert m1 != m3

        m4 = Machine(
            {
                TaskTarget.CPU: CPU_RANGE,
                TaskTarget.OMP: OMP_RANGE,
                TaskTarget.GPU: EMPTY_RANGE,
            }
        )
        assert m1 == m4

    @pytest.mark.parametrize("n", range(1, len(RANGES) + 1))
    def test_preferred_target(self, n: int) -> None:
        m = Machine(dict(zip(TARGETS[:n], RANGES[:n])))
        assert m.preferred_target == TARGETS[n - 1]
        assert len(m) == len(RANGES[n - 1])

    @pytest.mark.parametrize("n", range(1, len(RANGES) + 1))
    def test_valid_targets(self, n: int) -> None:
        m = Machine(dict(zip(TARGETS[:n], RANGES[:n])))
        assert set(m.valid_targets) == set(TARGETS[:n])

    def test_get_processor_range(self) -> None:
        m = Machine({TaskTarget.CPU: CPU_RANGE, TaskTarget.OMP: OMP_RANGE})
        assert m.get_processor_range(TaskTarget.CPU) == CPU_RANGE
        assert m.get_processor_range(TaskTarget.OMP) == OMP_RANGE
        assert m.get_processor_range() == OMP_RANGE
        assert len(m.get_processor_range(TaskTarget.GPU)) == 0

    def test_get_node_range(self) -> None:
        m = Machine(dict(zip(TARGETS, RANGES)))
        assert m.get_node_range(TaskTarget.CPU) == (0, 1)
        assert m.get_node_range(TaskTarget.OMP) == (0, 2)
        assert m.get_node_range(TaskTarget.GPU) == (1, 2)
        assert m.get_node_range() == m.get_node_range(TaskTarget.GPU)

    def test_only(self) -> None:
        m = Machine(dict(zip(TARGETS, RANGES)))
        gpu = TaskTarget.GPU
        cpu = TaskTarget.CPU
        omp = TaskTarget.OMP
        assert len(m.only(gpu)) == len(GPU_RANGE)
        assert m.only(gpu).only(gpu) == m.only(gpu)
        assert len(m.only(gpu).get_processor_range(cpu)) == 0
        assert m.only(gpu).get_processor_range(gpu) == GPU_RANGE
        assert len(m.only([gpu, cpu])) == len(GPU_RANGE)
        assert len(m.only([gpu, cpu]).only(gpu)) == len(GPU_RANGE)
        assert len(m.only([gpu, cpu]).only(cpu)) == len(CPU_RANGE)
        assert len(m.only([gpu, cpu]).only(omp)) == 0

    def test_count(self) -> None:
        m = Machine(dict(zip(TARGETS, RANGES)))
        assert m.count(TaskTarget.CPU) == len(CPU_RANGE)
        assert m.count(TaskTarget.OMP) == len(OMP_RANGE)
        assert m.count(TaskTarget.GPU) == len(GPU_RANGE)

    def test_get_item(self) -> None:
        m = Machine(dict(zip(TARGETS, RANGES)))
        assert m[TaskTarget.GPU] == Machine({TaskTarget.GPU: GPU_RANGE})
        assert m[ProcessorSlice(TaskTarget.GPU, slice(1, 2))] == Machine(
            {TaskTarget.GPU: GPU_RANGE[1:2]}
        )

        m = m.only(TaskTarget.GPU)
        assert m[4] == Machine({TaskTarget.GPU: GPU_RANGE[4]})
        assert m[4:] == Machine({TaskTarget.GPU: GPU_RANGE[4:]})
        assert m[:5] == Machine({TaskTarget.GPU: GPU_RANGE[:5]})

    def test_intersection(self) -> None:
        m1 = Machine({TaskTarget.CPU: CPU_RANGE, TaskTarget.OMP: OMP_RANGE})
        m2 = Machine({TaskTarget.OMP: OMP_RANGE, TaskTarget.GPU: GPU_RANGE})
        assert m1 & m2 == Machine({TaskTarget.OMP: OMP_RANGE})

        m1 = Machine({TaskTarget.CPU: CPU_RANGE})
        m2 = Machine({TaskTarget.OMP: OMP_RANGE})
        assert (m1 & m2).empty

    def test_empty(self) -> None:
        assert Machine().empty
        assert Machine({TaskTarget.GPU: EMPTY_RANGE}).empty
        assert not Machine(dict(zip(TARGETS, RANGES))).empty
        assert Machine().valid_targets == tuple()
        assert Machine({TaskTarget.GPU: EMPTY_RANGE}).valid_targets == tuple()

    def test_idempotent_scopes(self) -> None:
        from legate.core import get_machine

        machine = get_machine()
        with machine:
            assert machine == get_machine()
            machine = get_machine()
            with machine:
                assert machine == get_machine()

    def test_empty_scope(self) -> None:
        from legate.core import get_machine

        machine = get_machine()
        rng = machine.get_processor_range()
        empty_rng = ProcessorRange.create(
            low=1, high=0, per_node_count=rng.per_node_count
        )
        err_msg = "Empty machines cannot be used for resource scoping"
        with pytest.raises(EmptyMachineError, match=err_msg):
            with Machine({machine.preferred_target: empty_rng}):
                pass

    def test_set_machine_twice(self) -> None:
        from legate.core import get_machine

        machine = get_machine()
        with machine:
            err_msg = "Each machine can be set only once to the scope"
            with pytest.raises(ValueError, match=err_msg):
                with machine:
                    pass


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
