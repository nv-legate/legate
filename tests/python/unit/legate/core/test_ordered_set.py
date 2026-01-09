# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from legate.core.utils import OrderedSet


class TestOrderedSet:
    def test_empty(self) -> None:
        s: OrderedSet[int] = OrderedSet()
        assert len(s) == 0

    def test_from_tuple(self) -> None:
        t = tuple(range(10))
        s = OrderedSet(t)
        for idx, item in enumerate(s):
            assert item == t[idx]

    def test_from_dict(self) -> None:
        d = dict.fromkeys(range(10), "foo")
        s = OrderedSet(d.items())
        assert tuple(enumerate(s)) == tuple(enumerate(d.items()))

    def test_add_from_tuple(self) -> None:
        t = tuple(range(10))
        s: OrderedSet[int] = OrderedSet()
        for i in t:
            s.add(i)
        assert tuple(enumerate(s)) == tuple(enumerate(t))

    def test_update_from_tuple(self) -> None:
        t = tuple(range(10))
        s: OrderedSet[int] = OrderedSet()
        s.update(t)
        assert tuple(enumerate(s)) == tuple(enumerate(t))

    def test_discard(self) -> None:
        d = dict.fromkeys(range(10), "foo")
        s = OrderedSet(d.items())
        s.discard((5, d[5]))
        d.pop(5)
        assert tuple(enumerate(s)) == tuple(enumerate(d.items()))

    def test_remove(self) -> None:
        t1 = tuple(range(10))
        t2 = tuple(range(5, 10))
        s1 = OrderedSet(t1)
        s2 = OrderedSet(t2)
        s1 = s1.remove_all(s2)
        assert tuple(enumerate(s1)) == tuple(enumerate(t1[:5]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
