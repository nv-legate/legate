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


class BaseError(Exception):
    r"""Base exception"""


class UnsatisfiableConfigurationError(BaseError):
    r"""An error as a result of user configuration that cannot be satisfied.
    For example, the user has requested X, but it does not work. Or the user
    has told us to look for Y in Z, but we did not find it there.
    """


class CMakeConfigureError(BaseError):
    r"""An error as a result of CMake failing."""


class LengthError(BaseError):
    r"""An exception to signify an object that is not of the right length"""


class WrongOrderError(BaseError):
    r"""An error raised when an operation is performed in the wrong order, e.g.
    accessing an object before it has been setup, or retrieving a resource
    before it has been registered.
    """
