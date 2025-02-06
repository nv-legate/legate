# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import re

import pytest

from legate.core import TaskInfo, VariantCode, get_legate_runtime


class TestTaskInfo:
    def test_new_task_info_properties(self) -> None:
        tid = get_legate_runtime().core_library.get_new_task_id()
        variant = VariantCode.CPU
        tname = "foo"
        tinfo = TaskInfo.from_variants(tid, tname, [(variant, print)])
        assert tinfo.name == tname
        assert (
            str(tinfo) == f"TaskInfo(name: {tname}, variants: {variant.name})"
        )

    def test_has_variant(self) -> None:
        tid = get_legate_runtime().core_library.get_new_task_id()
        variant = VariantCode.CPU
        tinfo = TaskInfo.from_variants(tid, "foo", [(variant, print)])
        assert tinfo.has_variant(variant)
        for v in VariantCode:
            if v == variant:
                continue
            assert not tinfo.has_variant(v)


class TestTaskInfoErrors:
    def test_from_empty_variants(self) -> None:
        tid = get_legate_runtime().core_library.get_new_task_id()
        msg = "Variants must not be empty"
        with pytest.raises(ValueError, match=msg):
            TaskInfo.from_variants(tid, "foo", [])

    def test_register_non_callable(self) -> None:
        tid = get_legate_runtime().core_library.get_new_task_id()
        variant_kind = VariantCode.CPU
        fn = "foo"
        msg = re.escape(
            f"Variant function ({fn}) for variant kind "
            f"{variant_kind} is not callable"
        )
        with pytest.raises(TypeError, match=msg):
            TaskInfo.from_variants(
                tid,
                "foo",
                [(variant_kind, fn)],  # type: ignore[list-item]
            )

    def test_add_unknown_variants(self) -> None:
        tid = get_legate_runtime().core_library.get_new_task_id()
        variant_kind = 12345
        msg = f"{variant_kind} is not a valid VariantCode"
        with pytest.raises(ValueError, match=msg):
            TaskInfo.from_variants(
                tid,
                "foo",
                [(variant_kind, print)],  # type: ignore[list-item]
            )

    def test_add_existing_variant(self) -> None:
        tid = get_legate_runtime().core_library.get_new_task_id()
        variant_kind = VariantCode.CPU
        fn = print
        tinfo = TaskInfo.from_variants(tid, "foo", [(variant_kind, fn)])
        msg = re.escape(
            f"Already added callback ({fn}) for 1 variant (local id: {tid})"
        )
        with pytest.raises(RuntimeError, match=msg):
            tinfo.add_variant(variant_kind, fn)

    def test_no_registered_variants(self) -> None:
        runtime = get_legate_runtime()
        tid = runtime.core_library.get_new_task_id()
        tinfo = TaskInfo.from_variants(tid, "foo", [(VariantCode.CPU, print)])
        runtime.core_library.register_task(tinfo)
        msg = re.escape(f"Task (local id: {tid}) has no variants!")
        with pytest.raises(RuntimeError, match=msg):
            runtime.core_library.register_task(tinfo)

    def test_register_existing_task_id(self) -> None:
        runtime = get_legate_runtime()
        tid = runtime.core_library.get_new_task_id()
        tinfo = TaskInfo.from_variants(
            tid, "test_register_existing_task_id", [(VariantCode.CPU, print)]
        )
        runtime.core_library.register_task(tinfo)
        tinfo = TaskInfo.from_variants(
            tid, "test_register_existing_task_id", [(VariantCode.CPU, print)]
        )
        msg = f"Task {tid} already exists in library legate.core"
        with pytest.raises(ValueError, match=msg):
            runtime.core_library.register_task(tinfo)
