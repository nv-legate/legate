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
from collective import (
    collective_test,
    collective_test_matvec,
    create_int64_store,
)

SHAPE = (
    100,
    10,
)
TILE = (
    10,
    10,
)


def test_no_collective() -> None:
    store = create_int64_store(shape=(100, 10))
    collective_test(store, SHAPE, TILE)


def test_broadcast() -> None:
    store = create_int64_store(shape=(1, 10))
    collective_test(store, SHAPE, TILE)


def test_overlap() -> None:
    a = create_int64_store(
        shape=(
            100,
            100,
        )
    )
    b = create_int64_store(shape=(100))
    c = create_int64_store(shape=(100))
    collective_test_matvec(a, b, c)


def test_transpose() -> None:
    a = create_int64_store(shape=(200,))

    a = a.slice(0, slice(None, 100))
    a = a.promote(0, 100)
    a = a.transpose(
        [
            1,
            0,
        ]
    )
    shape = (
        100,
        100,
    )

    collective_test(a, shape, TILE)


def test_project() -> None:
    shape = (
        100,
        100,
    )
    a = create_int64_store(shape)

    a = a.promote(0, 100)
    a = a.project(0, 1)
    a = a.promote(1, 100)
    a = a.project(1, 1)
    collective_test(a, shape, TILE)


def test_2_promotions() -> None:
    a = create_int64_store(shape=(1,))

    a = a.promote(0, 100)
    a = a.promote(0, 100)
    a = a.project(1, 1)
    shape = (
        100,
        100,
    )
    collective_test(a, shape, TILE)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
