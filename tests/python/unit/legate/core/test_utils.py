# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from legate.core.utils import Annotation


class TestAnnotation:
    @pytest.mark.parametrize(
        "pairs", [{}, {"key1": 1}, {"key1": "value1", 2: "value2"}]
    )
    def test_context_manager_basic(self, pairs: dict[str, str]) -> None:
        ann = Annotation(pairs)

        with ann as ctx:
            # this is currently no-op, asserting is None here as a reminder
            # to update the test when it's actually implemented.
            assert ctx is None

        assert ann._pairs == pairs


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
