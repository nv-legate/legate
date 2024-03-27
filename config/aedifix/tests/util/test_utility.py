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

import sys

import pytest

from ...util.utility import (
    deduplicate_command_line_args,
    prune_command_line_args,
)


class TestUtility:
    def test_prune_command_line_args_empty(self) -> None:
        argv = [
            "--foo",
            "--bar=1",
            "--baz",
            "0",
            "1",
            "asdasd",
            "--remain",
            "--foo=yes",
            "--bar",
            "1",
        ]

        new_argv = prune_command_line_args(argv, set())
        assert new_argv == argv

    def test_prune_command_line_args(self) -> None:
        to_remove = {"--foo", "--bar", "--baz"}
        argv = [
            "--foo",
            "--bar=1",
            "--baz",
            "0",
            "1",
            "asdasd",
            "--remain",
            "--foo=yes",
            "--bar",
            "1",
        ]

        new_argv = prune_command_line_args(argv, to_remove)
        assert new_argv == ["--remain"]

    def test_prune_command_line_args_bad(self) -> None:
        bad_remove = {"asdasd", "asdau999"}

        with pytest.raises(
            ValueError, match=r"Argument '.*' must start with '-'"
        ):
            prune_command_line_args([], bad_remove)

    def test_deduplicate_command_line_args(self) -> None:
        argv = [
            "--foo=1",
            "--foo",
            "--foo=45",
            "--hello",
            "world",
            "--foo",
            "2",
            "--bar",
            "--baz=17",
        ]
        new_argv = deduplicate_command_line_args(argv)
        assert new_argv == [
            "--foo",
            "2",
            "--hello",
            "world",
            "--bar",
            "--baz=17",
        ]

    def test_deduplicate_command_line_args_empty(self) -> None:
        new_argv = deduplicate_command_line_args([])
        assert new_argv == []

    def test_deduplicate_command_line_args_positional_arg(self) -> None:
        new_argv = deduplicate_command_line_args(["foo", "--bar", "--foo"])
        assert new_argv == ["foo", "--bar", "--foo"]


if __name__ == "__main__":
    sys.exit(pytest.main())
