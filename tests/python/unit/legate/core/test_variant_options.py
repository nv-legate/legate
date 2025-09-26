# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from legate.core import VariantOptions


class TestVariantOptions:
    def test_basic(self) -> None:
        options = VariantOptions()

        assert options.concurrent is False
        assert options.has_allocations is False
        assert options.elide_device_ctx_sync is False
        assert options.has_side_effect is False
        assert options.may_throw_exception is False
        assert options.communicators == ()

    def test_construct(self) -> None:
        concurrent = True
        has_allocations = False
        elide_device_ctx_sync = True
        has_side_effect = False
        may_throw_exception = True
        communicators = ["comm1", "comm2"]

        options = VariantOptions(
            concurrent=concurrent,
            has_allocations=has_allocations,
            elide_device_ctx_sync=elide_device_ctx_sync,
            has_side_effect=has_side_effect,
            may_throw_exception=may_throw_exception,
            communicators=communicators,
        )

        assert options.concurrent == concurrent
        assert options.has_allocations == has_allocations
        assert options.elide_device_ctx_sync == elide_device_ctx_sync
        assert options.has_side_effect == has_side_effect
        assert options.may_throw_exception == may_throw_exception
        assert options.communicators == tuple(communicators)

    def test_setters(self) -> None:
        options = VariantOptions()

        concurrent = True
        has_allocations = False
        elide_device_ctx_sync = True
        has_side_effect = False
        may_throw_exception = True
        communicators = ["comm1", "comm2"]

        options.concurrent = concurrent
        options.has_allocations = has_allocations
        options.elide_device_ctx_sync = elide_device_ctx_sync
        options.has_side_effect = has_side_effect
        options.may_throw_exception = may_throw_exception
        options.communicators = communicators

        assert options.concurrent == concurrent
        assert options.has_allocations == has_allocations
        assert options.elide_device_ctx_sync == elide_device_ctx_sync
        assert options.has_side_effect == has_side_effect
        assert options.may_throw_exception == may_throw_exception
        assert options.communicators == tuple(communicators)

    def test_communicators_property(self) -> None:
        options = VariantOptions()

        assert options.communicators == ()

        communicators = ("comm1", "comm2")
        options.communicators = communicators
        assert options.communicators == communicators

        options.communicators = []
        assert options.communicators == ()

        options.communicators = communicators
        # Test that re-settting them to the values does in fact re-set them
        assert options.communicators == tuple(communicators)

        options.communicators = ()
        assert options.communicators == ()
