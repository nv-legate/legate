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

from typing import Optional

from ..._lib.data.physical_array cimport PhysicalArray
from ..._lib.data.physical_store cimport PhysicalStore
from ..._lib.mapping.mapping cimport TaskTarget
from ..._lib.partitioning.constraint cimport ConstraintProxy
from ..._lib.task.task_context cimport TaskContext

ctypedef dict[str, type] SignatureMapping
ctypedef tuple[str, ...] ParamList
ctypedef str VariantKind  # Literal["cpu", "gpu", "omp"]
ctypedef tuple[VariantKind, ...] VariantList
# Would use UserFunction | None below, but get:
#
# legate.core.internal/legate/core/_ext/task/type.pxd:26:39: Compiler crash in
# OptimizeBuiltinCalls
#
#     ModuleNode.body = StatListNode(type.pxd:11:0)
#     StatListNode.stats[11] = CTypeDefNode(type.pxd:26:0,
#         in_pxd = True,
#         visibility = 'private')
#     CTypeDefNode.base_type = TemplatedTypeNode(type.pxd:26:13,
#         is_templated_type_node = True)
#     TemplatedTypeNode.positional_args[1] = IntBinopNode(type.pxd:26:39,
#         infix = True,
#         operator = '|',
#         result_is_used = True,
#         use_managed_ref = True)
#
#     Compiler crash traceback from this point on:
#       File "/path/to/python3.11/site-packages/Cython/Compiler/Visitor.py",
#       line 182, in _visit
#         return handler_method(obj)
#                ^^^^^^^^^^^^^^^^^^^
#       File "/path/to/python3.11/site-packages/Cython/Compiler/Visitor.py",
#       line 545, in visit_BinopNode
#         return self._visit_binop_node(node)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       File "/path/to/python3.11/site-packages/Cython/Compiler/Visitor.py",
#       line 559, in _visit_binop_node
#         if obj_type.is_builtin_type:
#            ^^^^^^^^^^^^^^^^^^^^^^^^
#     AttributeError: 'NoneType' object has no attribute 'is_builtin_type'
#
# ...oops?
ctypedef dict[TaskTarget, Optional[UserFunction]] VariantMapping

cdef class InputStore(PhysicalStore):
    pass


cdef class OutputStore(PhysicalStore):
    pass

cdef class ReductionStore(PhysicalStore):
    pass


cdef class InputArray(PhysicalArray):
    pass


cdef class OutputArray(PhysicalArray):
    pass

ctypedef tuple[ConstraintProxy, ...] ConstraintSet
