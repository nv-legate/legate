/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DOXYGEN

#include <legate_defines.h>  // IWYU pragma: keep

#include <legate/utilities/macros.h>  // IWYU pragma: keep

extern "C" {

// NOLINTBEGIN
const char* __asan_default_options()
{
  return "check_initialization_order=1:"
         "detect_stack_use_after_return=1:"
         "alloc_dealloc_mismatch=1:"
         "abort_on_error=1:"
         "strict_string_checks=1:"
         "color=always:"
         "detect_odr_violation=2:"
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
         "protect_shadow_gap=0:"
#endif
#if LEGATE_DEFINED(LEGATE_MACOS)
         "detect_leaks=0:"
#endif
         // note trailing ":", this is so that user may write ASAN_OPTIONS+="foo:bar:baz"
         "symbolize=1:";
}

const char* __ubsan_default_options()
{
  return "halt_on_error=1:"
         "print_stacktrace=1:"
         "report_error_type=1:";
}

const char* __ubsan_default_suppressions()
{
  return "vptr:Legion::Internal::RemoteContext::RemoteContext\n"
         "vptr:Legion::Internal::Runtime::find_or_request_equivalence_set\n"
         "enum:Realm::Cuda::CudaModule::create_module\n";
}

const char* __lsan_default_suppressions()
{
  return
    // This is the only symbol that Legate should ever leak from
    "leak:intentionally_leak*\n"
    // Legion and Realm leak like faucets
    "leak:liblegion.*\n"
    "leak:librealm.*\n"
    "leak:liblegion-legate.*\n"
    "leak:librealm-legate.*\n"
    // Because we didn 't build python with ASAN (in fact, we don' t build python at all),
    // these raise a bunch of(possibly) false positives related to interned Python strings
    // and Python module initialization.Maybe they are real errors, maybe not, but in any
    // case these are totally out of our control.
    "leak:CRYPTO_malloc\n"
    "leak:Py_initialize\n"
    "leak:Py_InitializeFromConfig\n"
    "leak:PyBytes_Resize\n"
    "leak:PyMem_RawCalloc\n"
    "leak:PyMem_RawMalloc\n"
    "leak:PyMem_RawRealloc\n"
    "leak:PyObject_Calloc\n"
    "leak:PyObject_GC_Resize\n"
    "leak:PyObject_Malloc\n"
    "leak:PyObject_Realloc\n"
    "leak:PyThread_allocate\n"
    "leak:_PyUnicodeWriter_Finish\n"
    "leak:list_resize\n"
    "leak:resize_buffer\n"
    "leak:resize_compact\n"
    "leak:unicode_resize\n"
    "leak:_PyList_AppendTakeRefListResize\n"
    // Numpy
    "leak:_multiarray_umath.*\n"
    "leak:numpy\n"
    // Cython
    "leak:__Pyx_CyFunction_Vectorcall_FASTCALL_KEYWORDS\n"
    "leak:__Pyx_CyFunction_Vectorcall*\n"
    "leak:PyObject_Vectorcall\n"
#if LEGATE_DEFINED(LEGATE_MACOS)
    "leak:CoreFoundation\n"
    "leak:libobjc.*\n"
#endif
    "leak:site-packages/cupy/cuda/memory.cpython\n"
    // UCC
    "leak:ucc_tl_nccl_team_t_init\n"
    "leak:kh_init_tl_cuda_ep_hash\n"
    "leak:ucc_tl_cuda_context_t_init\n"
    "leak:ucc_cl_lib_config_read\n"
    "leak:ucc_mc_init\n"
    "leak:kh_init_ucc_mc_cuda_resources_hash\n"
    "leak:ucc_mc_cuda_init\n"
    "leak:ucs_*\n"
    "leak:ucc_*\n"
    // Verbs
    "leak:libibverbs.*\n";
}

const char* __tsan_default_options()
{
  return "halt_on_error=1:"
         "second_deadlock_stack=1:"
         "symbolize=1:"
         "detect_deadlocks=1:";
}

const char* __tsan_default_suppressions()
{
  return "race:Legion::Internal::MemoryManager::create_eager_instance\n"
         "race:Legion::Internal::Operation::perform_registration\n";
}

// NOLINTEND

}  // extern "C"
#endif
