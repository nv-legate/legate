# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

class Unconstructable:
    # Technically this is a "NoReturn" function, but mypy really does not like
    # this. If you make this NoReturn then it basically makes any derived class
    # unusable, since mypy will just consider any and all uses of the class as
    # calling the ctor. For example:
    #
    # tests/unit/legate/core/test_logicalarray.py: note: In member "test_basic"
    # of class "TestFromRawHandle":
    # tests/unit/legate/core/test_logicalarray.py:420:16: error:
    # "Callable[[VarArg(Any), KwArg(Any)], Never]" has no attribute
    # "from_raw_handle" [attr-defined]
    #             arr2 = LogicalArray.from_raw_handle(arr.raw_handle)
    #                    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # tests/unit/legate/core/util/task_util.py: note: In function
    # "single_input": tests/unit/legate/core/util/task_util.py:258:26: error:
    # Argument 2 to "assert_isinstance" has incompatible type
    # "Callable[[VarArg(Any), KwArg(Any)], Never]"; expected "type[Never]"
    # [arg-type]
    #         assert_isinstance(a, PhysicalStore)
    #                              ^~~~~~~~~~~~~
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
