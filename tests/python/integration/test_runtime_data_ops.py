# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator
import numpy as np

import pytest

from legate.core import LEGATE_MAX_DIM, Scalar, get_legate_runtime, types as ty

from .utils import utils
from .utils.data import ARRAY_TYPES, SCALAR_VALS


class TestStoreOps:
    @pytest.mark.parametrize(
        ("dtype", "val"),
        tuple(zip(ARRAY_TYPES, SCALAR_VALS, strict=True)),
        ids=str,
    )
    @pytest.mark.parametrize("create", [True, False])
    def test_issue_fill_scalar(
        self, dtype: ty.Type, val: Any, create: bool
    ) -> None:
        shape = range(1, LEGATE_MAX_DIM + 1)
        runtime = get_legate_runtime()
        store = runtime.create_store(dtype, shape)
        scalar = Scalar(val, dtype) if create else val
        runtime.issue_fill(store, scalar)
        arr = np.asarray(store.get_physical_store())
        assert (arr == scalar).all()
        assert (arr == val).all()

    @pytest.mark.parametrize(
        ("dtype", "val"),
        tuple(zip(ARRAY_TYPES, SCALAR_VALS, strict=True)),
        ids=str,
    )
    def test_issue_fill_store(self, dtype: ty.Type, val: Any) -> None:
        runtime = get_legate_runtime()
        arr, store = utils.create_np_array_and_store(
            dtype, tuple(range(1, LEGATE_MAX_DIM + 1))
        )
        val_store = runtime.create_store_from_scalar(Scalar(val, dtype))
        runtime.issue_fill(store, val_store)
        assert (arr == val).all()


class TestStoreOpsErrors:
    def test_fill_non_nullable(self) -> None:
        runtime = get_legate_runtime()
        msg = "Fill value and target must have the same type"
        store = runtime.create_store(ty.float64, (1, 1))
        with pytest.raises(ValueError, match=msg):
            runtime.issue_fill(store, None)

    @pytest.fixture
    def preserve_seed_state(self) -> Generator[None, Any, Any]:
        state = np.random.get_state()
        yield
        np.random.set_state(state)

    @pytest.mark.usefixtures("preserve_seed_state")
    def test_rand_seed_warn(self) -> None:
        # Not exactly runtime, just not worth a new file.
        msg = """
        Seeding the random number generator with a non-constant value
        inside Legate can lead to undefined behavior and/or errors when
        the program is executed with multiple ranks."""
        with pytest.warns(Warning, match=msg):
            np.random.seed()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
