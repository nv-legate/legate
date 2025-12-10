/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/redop/detail/register.h>

#include <legate_defines.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/redop/complex.h>
#include <legate/redop/half.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/detail/types.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

#include <realm/cuda/cuda_module.h>

#include <optional>
#include <stdexcept>
#include <vector>

namespace legate::detail {

namespace {

// TODO(jfaibussowit)
// Use this to iterate over all the reduction operators and register them directly with
// Legate. Currently this macro is unused.
#define LEGATE_FOREACH_REDOP(__op__, ...)              \
  do {                                                 \
    LEGATE_FOREACH_BOOL_REDOP(__op__, __VA_ARGS__);    \
    LEGATE_FOREACH_INT8_REDOP(__op__, __VA_ARGS__);    \
    LEGATE_FOREACH_INT16_REDOP(__op__, __VA_ARGS__);   \
    LEGATE_FOREACH_INT32_REDOP(__op__, __VA_ARGS__);   \
    LEGATE_FOREACH_INT64_REDOP(__op__, __VA_ARGS__);   \
    LEGATE_FOREACH_UINT8_REDOP(__op__, __VA_ARGS__);   \
    LEGATE_FOREACH_UINT16_REDOP(__op__, __VA_ARGS__);  \
    LEGATE_FOREACH_UINT32_REDOP(__op__, __VA_ARGS__);  \
    LEGATE_FOREACH_UINT64_REDOP(__op__, __VA_ARGS__);  \
    LEGATE_FOREACH_FLOAT16_REDOP(__op__, __VA_ARGS__); \
    LEGATE_FOREACH_FLOAT32_REDOP(__op__, __VA_ARGS__); \
    LEGATE_FOREACH_FLOAT64_REDOP(__op__, __VA_ARGS__); \
    LEGATE_FOREACH_COMPLEX_REDOP(__op__, __VA_ARGS__); \
  } while (0)

#define LEGATE_FOREACH_SPECIALIZED_REDOP(__op__, ...)              \
  do {                                                             \
    LEGATE_FOREACH_SPECIALIZED_FLOAT16_REDOP(__op__, __VA_ARGS__); \
    LEGATE_FOREACH_SPECIALIZED_COMPLEX_REDOP(__op__, __VA_ARGS__); \
  } while (0)

#define LEGATE_REGISTER_CUDA_REDOP(T, processor, mod_manager_ptr, desc_vec_ptr) \
  do {                                                                          \
    auto& desc = (desc_vec_ptr)->emplace_back();                                \
                                                                                \
    try {                                                                       \
      desc.proc = processor;                                                    \
      T::fill_redop_desc(mod_manager_ptr, &desc);                               \
    } catch (...) {                                                             \
      (desc_vec_ptr)->pop_back();                                               \
      throw;                                                                    \
    }                                                                           \
  } while (0)

/**
 * @brief Retrieves the CUDA context for a given processor.
 *
 * Unfortunately, Realm does not compile any of the CUDA module code unless it is configured
 * with CUDA support, so while all the CUDA module code would compile, it would produce link
 * errors down the line.
 *
 * So we make these shim functions to stub them in more nicely.
 *
 * @param cuda Pointer to the Realm CUDA module.
 * @param p Processor for which to get the CUDA context.
 *
 * @return Optional CUDA context.
 */
[[nodiscard]] std::optional<CUcontext> realm_get_cuda_context(const Realm::Cuda::CudaModule* cuda,
                                                              const Processor& p)
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    CUcontext ctx{};

    if (cuda->get_cuda_context(p, &ctx)) {
      return ctx;
    }
  }
  return std::nullopt;
}

/**
 * @brief Registers CUDA reduction operators with a Realm CUDA module.
 *
 * Attempts to register all descriptors with the given CUDA module. Throws an exception if
 * registration fails. Waits for the registration event to complete.
 *
 * @param cuda Pointer to the Realm CUDA module.
 * @param descs Span of `CudaRedOpDesc` to register.
 */
void realm_register_cuda_reductions(Realm::Cuda::CudaModule* cuda,
                                    Span<const Realm::Cuda::CudaRedOpDesc> descs)
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    auto e = Realm::Event::NO_EVENT;

    if (!cuda->register_reduction(e, descs.data(), descs.size())) {
      throw TracedException<std::runtime_error>{
        "Failed to register reductions directly with Realm CUDA module"};
    }
    e.wait();
  }
}

/**
 * @brief Get the Realm CUDA module, if it exists.
 *
 * @return The CUDA module, or std::nullopt if it doesn't exist.
 */
[[nodiscard]] std::optional<Realm::Cuda::CudaModule*> realm_get_cuda_module()
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    auto* const mod = Realm::Runtime::get_runtime().get_module<Realm::Cuda::CudaModule>("cuda");

    if (mod) {
      // The CUDA module might not exist if the user has disabled them via LEGATE_CONFIG (via
      // --gpus 0).
      return mod;
    }
  }
  return std::nullopt;
}

/**
 * @brief Creates CUDA reduction operator descriptors for all GPUs.
 *
 * Queries all TOC (GPU) processors and generates `CudaRedOpDesc` entries for each registered
 * reduction operator, using the given CUDA module.
 *
 * @param cuda Pointer to the CUDA module.
 * @return A vector of `CudaRedOpDesc` describing the CUDA reduction operators.
 */
[[nodiscard]] std::vector<Realm::Cuda::CudaRedOpDesc> create_cuda_redop_descriptors(
  const Realm::Cuda::CudaModule* cuda)
{
  auto&& mod_manager = Runtime::get_runtime().get_cuda_module_manager();

  const auto pq = Realm::Machine::ProcessorQuery{Realm::Machine::get_machine()}.only_kind(
    Realm::Processor::TOC_PROC);
  std::vector<Realm::Cuda::CudaRedOpDesc> descs;

  // No point in reserving descs. We push_back() an entry for every redop, and we currently
  // don't have a way of enumerating exactly how many redops there are. Furthermore, if the
  // processor in our query is not a GPU, then we would skip that entire section. So any guess
  // we make here would be wrong.
  for (auto&& p : pq) {
    if (const auto ctx = realm_get_cuda_context(cuda, p); ctx.has_value()) {
      const auto _ = cuda::detail::AutoCUDAContext{*ctx};

      LEGATE_FOREACH_SPECIALIZED_REDOP(LEGATE_REGISTER_CUDA_REDOP, p, &mod_manager, &descs);
    }
  }
  return descs;
}

/**
 * @brief Records a reduction operator for a specific type.
 *
 * Registers the mapping between a reduction operation and its GlobalRedopID for the given
 * primitive type.
 *
 * @param op The reduction operation kind.
 * @param type_code The type code of the primitive type.
 */
void record(ReductionOpKind op, Type::Code type_code)
{
  PrimitiveType{type_code}.record_reduction_operator(to_underlying(op),
                                                     builtin_redop_id(op, type_code));
}

/**
 * @brief Records a reduction operation for all integer types.
 *
 * @param op The reduction operation kind to record.
 */
void record_int(ReductionOpKind op)
{
  record(op, Type::Code::BOOL);
  record(op, Type::Code::INT8);
  record(op, Type::Code::UINT8);
  record(op, Type::Code::INT16);
  record(op, Type::Code::UINT16);
  record(op, Type::Code::INT32);
  record(op, Type::Code::UINT32);
  record(op, Type::Code::INT64);
  record(op, Type::Code::UINT64);
}

/**
 * @brief Records a reduction operation for all floating point types.
 *
 * @param op The reduction operation kind to record.
 */
void record_float(ReductionOpKind op)
{
  record(op, Type::Code::FLOAT16);
  record(op, Type::Code::FLOAT32);
  record(op, Type::Code::FLOAT64);
}

/**
 * @brief Records a reduction operation for all complex types.
 *
 * @param op The reduction operation kind to record.
 */
void record_complex(ReductionOpKind op) { record(op, Type::Code::COMPLEX64); }

/**
 * @brief Records a reduction operation for all supported types.
 *
 * Invokes type-specific recorders (int, float, complex) for the given reduction kind.
 *
 * @param op The reduction operation kind to record.
 */
void record_all(ReductionOpKind op)
{
  record_int(op);
  record_float(op);
  record_complex(op);
}

}  // namespace

#define LEGATE_REGISTER_LEGION_REDOP(__T__, ...) \
  Legion::Runtime::register_reduction_op<__T__>(static_cast<Realm::ReductionOpID>(__T__::REDOP_ID))

void register_builtin_reduction_ops()
{
  LEGATE_FOREACH_SPECIALIZED_REDOP(LEGATE_REGISTER_LEGION_REDOP, );
  if (const auto cuda = realm_get_cuda_module(); cuda.has_value()) {
    const auto descs = create_cuda_redop_descriptors(*cuda);

    realm_register_cuda_reductions(*cuda, descs);
  }

  record_all(ReductionOpKind::ADD);
  record(ReductionOpKind::ADD, Type::Code::COMPLEX128);
  record_all(ReductionOpKind::MUL);

  record_int(ReductionOpKind::MAX);
  record_float(ReductionOpKind::MAX);

  record_int(ReductionOpKind::MIN);
  record_float(ReductionOpKind::MIN);

  record_int(ReductionOpKind::OR);
  record_int(ReductionOpKind::AND);
  record_int(ReductionOpKind::XOR);
}

}  // namespace legate::detail
