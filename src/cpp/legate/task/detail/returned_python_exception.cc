/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/returned_python_exception.h>

#include <legate/task/detail/exception.h>
#include <legate/task/detail/returned_exception.h>
#include <legate/task/detail/returned_exception_common.h>
#include <legate/utilities/detail/pack.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/machine.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace legate::detail {

ReturnedPythonException::Payload::Payload(std::size_t size,
                                          std::unique_ptr<std::byte[]> bytes,
                                          std::string m) noexcept
  : pkl_size{size}, pkl_bytes{std::move(bytes)}, msg{std::move(m)}
{
}

namespace {

[[nodiscard]] std::unique_ptr<std::byte[]> copy_bytes(Span<const std::byte> span)
{
  auto ret = std::unique_ptr<std::byte[]>{new std::byte[span.size()]};

  std::memcpy(ret.get(), span.data(), span.size());
  return ret;
}

}  // namespace

ReturnedPythonException::ReturnedPythonException(Span<const std::byte> pkl_span, std::string msg)
  : bytes_{make_internal_shared<Payload>(pkl_span.size(), copy_bytes(pkl_span), std::move(msg))}
{
}

ReturnedPythonException::ReturnedPythonException(const std::byte* pkl_buf,
                                                 std::size_t pkl_len,
                                                 std::string msg)
  : ReturnedPythonException{Span<const std::byte>{pkl_buf, pkl_len}, std::move(msg)}
{
}

void ReturnedPythonException::legion_serialize(void* buffer) const
{
  auto rem_cap = legion_buffer_size();

  std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, kind());

  const auto pkl_span = pickle();

  std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, pkl_span.size());
  std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, pkl_span.size(), pkl_span.data());

  const auto mess = message();

  std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, mess.size());
  std::ignore               = pack_buffer(buffer, rem_cap, mess.size(), mess.data());
}

void ReturnedPythonException::legion_deserialize(const void* buffer)
{
  // There is no information about the size of the buffer, nor can we know how much we need
  // until we unpack all of it. So we just lie and say we have infinite memory.
  auto rem_cap = std::numeric_limits<std::size_t>::max();
  ExceptionKind kind;

  std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &kind);
  LEGATE_ASSERT(kind == ExceptionKind::PYTHON);

  // This temporary is used in order to preserve strong exception guarantee
  auto tmp_bytes = make_internal_shared<Payload>();

  std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &tmp_bytes->pkl_size);
  // NOLINTNEXTLINE(readability-magic-numbers)
  static_assert(LEGATE_CPP_MIN_VERSION < 20, "Use make_unique_for_overwrite below");
  tmp_bytes->pkl_bytes.reset(new std::byte[tmp_bytes->pkl_size]);

  auto* const ptr = tmp_bytes->pkl_bytes.get();

  std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, tmp_bytes->pkl_size, &ptr);

  std::decay_t<decltype(tmp_bytes->msg)>::size_type mess_size{};

  std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &mess_size);
  tmp_bytes->msg.assign(static_cast<const char*>(buffer), mess_size);
  bytes_ = std::move(tmp_bytes);
}

ReturnValue ReturnedPythonException::pack() const
{
  const auto buffer_size = legion_buffer_size();

  if (buffer_size > ReturnedException::max_size()) {
    throw TracedException<std::runtime_error>{
      fmt::format("The size of raised exception ({}) exceeds the maximum number of exception ({}). "
                  "Please increase the value for LEGATE_MAX_EXCEPTION_SIZE.",
                  buffer_size,
                  ReturnedException::max_size())};
  }

  const auto mem_kind = find_memory_kind_for_executing_processor();
  auto buffer         = Legion::UntypedDeferredValue{buffer_size, mem_kind};
  const auto acc      = AccessorWO<std::int8_t, 1>{buffer, buffer_size, false};

  legion_serialize(acc.ptr(0));
  // No alignment for returned exceptions, as they are always memcpy-ed
  return {std::move(buffer), buffer_size, /*alignment=*/1};  // NOLINT(performance-move-const-arg)
}

std::string ReturnedPythonException::to_string() const
{
  const auto pkl_span = pickle();
  std::string ret;

  fmt::format_to(
    std::back_inserter(ret), "ReturnedPythonException(size = {}, bytes = ", pkl_span.size());
  for (auto&& i : pkl_span) {
    fmt::format_to(std::back_inserter(ret), "\\x{:0x}", static_cast<std::int32_t>(i));
  }
  fmt::format_to(std::back_inserter(ret), ", message = {})", message());
  return ret;
}

void ReturnedPythonException::throw_exception()
{
  if (!raised()) {
    throw TracedException<std::logic_error>{
      "Cannot throw TracedException<exception>, as this python exception object is empty"};
  }

  const auto pkl_size = bytes_->pkl_size;
  auto ptr            = SharedPtr<const std::byte[]>{std::move(bytes_->pkl_bytes)};

  // Point of no return, at this point ptr owns the pickle bytes...
  bytes_.reset();
  // Should not wrap this exception in a trace, it may already contain a traced exception.
  throw PythonTaskException{pkl_size, std::move(ptr)};  // legate-lint: no-trace
}

}  // namespace legate::detail
