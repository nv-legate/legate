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


import pytest

from legate.core import get_legate_runtime, types as ty


class Test_store_creation:
    def test_bound(self) -> None:
        runtime = get_legate_runtime()
        context = runtime.core_context
        store = context.create_store(ty.int64, shape=(4, 4))
        assert not store.unbound
        assert store.ndim == 2
        assert store.shape == (4, 4)
        assert store.type == ty.int64
        assert not store.transformed

    def test_unbound(self) -> None:
        runtime = get_legate_runtime()
        context = runtime.core_context
        store = context.create_store(ty.int64)
        assert store.unbound
        assert store.ndim == 1
        assert store.type == ty.int64
        assert not store.transformed
        with pytest.raises(ValueError):
            store.shape


class Test_store_valid_transform:
    def test_bound(self) -> None:
        runtime = get_legate_runtime()
        context = runtime.core_context
        store = context.create_store(ty.int64, shape=(4, 3))

        promoted = store.promote(0, 5)
        assert promoted.shape == (5, 4, 3)
        assert promoted.transformed

        projected = store.project(0, 1)
        assert projected.shape == (3,)
        assert projected.transformed

        sliced = store.slice(1, slice(1, 3))
        assert sliced.shape == (4, 2)
        assert sliced.transformed

        transposed = store.transpose((1, 0))
        assert transposed.shape == (3, 4)
        assert transposed.transformed

        delinearized = store.delinearize(0, (2, 2))
        assert delinearized.shape == (2, 2, 3)
        assert delinearized.transformed


class Test_store_invalid_transform:
    def test_bound(self) -> None:
        runtime = get_legate_runtime()
        context = runtime.core_context
        store = context.create_store(ty.int64, shape=(4, 3))

        with pytest.raises(ValueError):
            store.promote(3, 5)

        with pytest.raises(ValueError):
            store.promote(-3, 5)

        with pytest.raises(ValueError):
            store.project(2, 1)

        with pytest.raises(ValueError):
            store.project(-3, 1)

        with pytest.raises(ValueError):
            store.project(0, 4)

        with pytest.raises(ValueError):
            store.slice(2, slice(1, 3))

        with pytest.raises(NotImplementedError):
            store.slice(1, slice(1, 3, 2))

        with pytest.raises(ValueError):
            store.slice(1, slice(1, 4))

        with pytest.raises(ValueError):
            store.transpose((2,))

        with pytest.raises(ValueError):
            store.transpose((0, 0))

        with pytest.raises(ValueError):
            store.transpose((2, 0))

        with pytest.raises(ValueError):
            store.delinearize(2, (2, 3))

        with pytest.raises(ValueError):
            store.delinearize(0, (2, 3))

    def test_unbound(self) -> None:
        runtime = get_legate_runtime()
        context = runtime.core_context
        store = context.create_store(ty.int64)
        with pytest.raises(ValueError):
            store.promote(1, 1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
