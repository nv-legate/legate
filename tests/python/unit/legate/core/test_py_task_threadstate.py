# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from legate.core import get_legate_runtime
from legate.core._ext.task import (  # type: ignore[attr-defined]
    python_task as m,
)
from legate.core.task import task


class TestPyTaskThreadState:
    def test_maybe_cache_hit(self) -> None:
        @task
        def task_a() -> None:
            pass

        @task
        def task_b() -> None:
            pass

        # Ensure no prior tasks are running
        get_legate_runtime().issue_execution_fence(block=True)
        # There are 2 scenarios here:
        #
        # 1. We are the first tasks to *ever* be executed by Legate:
        #
        #    In this case, task_a() should see an empty cache, and create a new
        #    threadstate. On exit, it should then stash the threadstate in the
        #    cache. task_b() then runs, and finds the threadstate in the cache
        #    and reuses it. In total, we should get 1 cache miss, and 1 cache
        #    hit.
        #
        # 2. We are being run after other tasks have run:
        #
        #    task_a() should now see a cached threadstate. It should remove the
        #    threadstate from the cache, use it, then put it back where it
        #    found it. task_b() does the same. So we should have 0 cache
        #    misses, and 2 cache hits.
        ctr_init = m._TLS_CACHE_HIT_COUNTER
        task_a()
        get_legate_runtime().issue_execution_fence(block=True)
        ctr_after_a = m._TLS_CACHE_HIT_COUNTER
        task_b()
        get_legate_runtime().issue_execution_fence(block=True)
        ctr_after = m._TLS_CACHE_HIT_COUNTER
        if ctr_after_a == ctr_init:
            # Scenario 1
            assert ctr_init + 1 == ctr_after
        else:
            # Scenario 2
            assert ctr_init + 2 == ctr_after


if __name__ == "__main__":
    pytest.main()
