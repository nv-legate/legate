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

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    from ._legion import Future


class PendingException:
    def __init__(
        self,
        exn_types: list[type],
        future: Future,
        tb_repr: Optional[str] = None,
    ):
        self._exn_types = exn_types
        self._future = future
        self._tb_repr = tb_repr

    def raise_exception(self) -> None:
        buf = self._future.get_buffer()
        (raised,) = struct.unpack("?", buf[:1])
        if not raised:
            return
        (exn_index, error_size) = struct.unpack("iI", buf[1:9])
        error_message = buf[9 : 9 + error_size].decode()
        exn_type = self._exn_types[exn_index]
        exn_reraised = exn_type(error_message)
        if self._tb_repr is not None:
            error_message += "\n" + self._tb_repr[:-1]  # remove extra newline
        exn_original = exn_type(error_message)
        raise exn_reraised from exn_original
