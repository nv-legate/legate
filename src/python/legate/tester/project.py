# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities and helpers for implementing the Legate custom test runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..util.types import EnvDict
    from . import CustomTest, FeatureType

__all__ = ("Project",)


class Project:
    def skipped_examples(self) -> set[str]:
        """Paths to test files that should be skipped entirely in all stages.

        Client test scripts can override this method to return their own
        customizations.
        """
        return set()

    def custom_files(self) -> list[CustomTest]:
        """Customized configurations for specific test files.

        Each entry will result in the specified test file being run in the
        specified stage, with the given command line arguments appended
        (overriding default stage arguments). These files are run serially,
        after the sharded, parallelized tests.

        Client test scripts can override this method to return their own
        customizations
        """
        return []

    def stage_env(self, feature: FeatureType) -> EnvDict:  # noqa: ARG002
        """Extra environment variables for the project, based on the
        stage feature type.

        Client test scripts can override this method to return their own
        customizations
        """
        return {}
