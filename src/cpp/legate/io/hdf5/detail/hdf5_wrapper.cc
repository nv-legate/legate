/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/hdf5_wrapper.h>

#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <H5Dpublic.h>
#include <H5Epublic.h>
#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>
#include <H5public.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#if LEGATE_DEFINED(LEGATE_USE_HDF5_VFD_GDS)
#include <H5FDgds.h>
#else
// NOLINTBEGIN
#define H5Pset_fapl_gds(fapl_id, alignment, block_size, cbuf_size) \
  [&]() -> int {                                                   \
    static_cast<void>(fapl_id);                                    \
    static_cast<void>(alignment);                                  \
    static_cast<void>(block_size);                                 \
    static_cast<void>(cbuf_size);                                  \
    return 0;                                                      \
  }()
// NOLINTEND
#endif

namespace legate::io::hdf5::detail::wrapper {

HDF5MaybeLockGuard::HDF5MaybeLockGuard() : lock_{mutex_, std::defer_lock}
{
  static const auto need_lock = [] {
    bool ret = false;

    LEGATE_CHECK(H5open() >= 0);
    LEGATE_CHECK(H5is_library_threadsafe(&ret) >= 0);
    return !ret;
  }();

  if (need_lock) {
    lock_.lock();
  }
}

namespace {

[[nodiscard]] herr_t stack_walk(unsigned n, const H5E_error2_t* err_desc, void* client_data)
{
  constexpr auto ERR_BUF_SIZE = 128;
  auto err_type               = H5E_type_t{};
  auto minor_error_buf        = std::array<char, ERR_BUF_SIZE>{};
  auto major_error_buf        = std::array<char, ERR_BUF_SIZE>{};
  auto cls                    = std::array<char, ERR_BUF_SIZE>{};

  std::ignore = H5Eget_class_name(err_desc->cls_id, cls.data(), cls.size());
  std::ignore =
    H5Eget_msg(err_desc->maj_num, &err_type, major_error_buf.data(), major_error_buf.size());
  std::ignore =
    H5Eget_msg(err_desc->min_num, &err_type, minor_error_buf.data(), minor_error_buf.size());

  auto& string = *static_cast<std::string*>(client_data);

  fmt::format_to(std::back_inserter(string),
                 "#{} {}:{} ({}): {}\n"
                 "  class: {}\n"
                 "  major: {}\n"
                 "  minor: {}\n",
                 n,
                 err_desc->file_name,
                 err_desc->line,
                 err_desc->func_name,
                 err_desc->desc,
                 cls.data(),
                 major_error_buf.data(),
                 minor_error_buf.data());
  return 0;
}

[[noreturn]] void throw_hdf5_exception(const HDF5MaybeLockGuard&, std::string ret)
{
  const auto stack = H5Eget_current_stack();

  ret += '\n';

  std::ignore = H5Ewalk(stack, H5E_WALK_DOWNWARD, &stack_walk, &ret);
  std::ignore = H5Eclear(stack);

  throw legate::detail::TracedException<std::runtime_error>{std::move(ret)};
};

#define HDF5_CALL_NO_ERROR_PRINTING(...)                                            \
  [&] {                                                                             \
    void* __legate_hdf5_client_data__ = nullptr;                                    \
    H5E_auto2_t __legate_hdf5_func__  = nullptr;                                    \
                                                                                    \
    H5Eget_auto2(H5E_DEFAULT, &__legate_hdf5_func__, &__legate_hdf5_client_data__); \
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);                                    \
                                                                                    \
    const auto __legate_hdf5_call_ret__ = __VA_ARGS__;                              \
                                                                                    \
    H5Eset_auto2(H5E_DEFAULT, __legate_hdf5_func__, __legate_hdf5_client_data__);   \
    return __legate_hdf5_call_ret__;                                                \
  }()

// ==========================================================================================

[[nodiscard]] std::string h5i_get_name(const HDF5MaybeLockGuard&, hid_t cls)
{
  // Calling with name and size equal to 0 returns the size we need to allocate for the string.
  auto size = HDF5_CALL_NO_ERROR_PRINTING(H5Iget_name(cls, /* name */ nullptr, /* size */ 0));
  auto ret  = std::string{};

  // size <= 0 indicates cls has no name (but not error, I don't think).
  if (size > 0) {
    ++size;  // for null terminator
    ret.resize(static_cast<std::size_t>(size));
    HDF5_CALL_NO_ERROR_PRINTING(H5Iget_name(cls, ret.data(), static_cast<std::size_t>(size)));
  }
  return ret;
}

[[nodiscard]] hid_t h5i_inc_ref(const HDF5MaybeLockGuard& lock, hid_t cls)
{
  const auto ret = HDF5_CALL_NO_ERROR_PRINTING(H5Iinc_ref(cls));

  if (ret < 0) {
    throw_hdf5_exception(lock, "Failed to increment reference count of object");
  }
  return cls;
}

// ==========================================================================================

[[nodiscard]] hid_t h5f_open(const HDF5MaybeLockGuard& lock,
                             legate::detail::ZStringView filename,
                             std::uint32_t mode,
                             hid_t fapl_id)
{
  const auto file_id = HDF5_CALL_NO_ERROR_PRINTING(
    H5Fopen(filename.data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
            mode,
            fapl_id));

  if (file_id == H5I_INVALID_HID) {
    throw_hdf5_exception(lock, fmt::format("Failed to open file {}", filename));
  }
  return file_id;
}

[[nodiscard]] hid_t h5f_create(const HDF5MaybeLockGuard& lock,
                               legate::detail::ZStringView filename,
                               std::uint32_t mode,
                               hid_t fcpl_id,
                               hid_t fapl_id)
{
  const auto file_id = HDF5_CALL_NO_ERROR_PRINTING(
    H5Fcreate(filename.data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
              mode,
              fcpl_id,
              fapl_id));

  if (file_id == H5I_INVALID_HID) {
    throw_hdf5_exception(lock, fmt::format("Failed to create file {}", filename));
  }
  return file_id;
}

namespace nothrow {

void h5f_close(hid_t file) noexcept
{
  const auto _ = HDF5MaybeLockGuard{};

  LEGATE_CHECK(H5Fclose(file) >= 0);
}

}  // namespace nothrow

// ==========================================================================================

[[nodiscard]] hid_t h5s_create_simple(const HDF5MaybeLockGuard& lock,
                                      Span<const hsize_t> dims,
                                      Span<const hsize_t> maxdims)
{
  const hid_t space_id = HDF5_CALL_NO_ERROR_PRINTING(
    H5Screate_simple(static_cast<int>(dims.size()), dims.data(), maxdims.data()));

  if (space_id == H5I_INVALID_HID) {
    throw_hdf5_exception(lock, "Failed to create simple dataspace");
  }
  return space_id;
}

namespace nothrow {

void h5s_close(hid_t space) noexcept
{
  const auto _ = HDF5MaybeLockGuard{};

  LEGATE_CHECK(H5Sclose(space) >= 0);
}

}  // namespace nothrow

void h5s_select_hyperslab(const HDF5MaybeLockGuard& lock,
                          hid_t space_id,
                          H5S_seloper_t mode,
                          Span<const hsize_t> start,
                          Span<const hsize_t> stride,
                          Span<const hsize_t> count,
                          Span<const hsize_t> block)
{
  const auto err = HDF5_CALL_NO_ERROR_PRINTING(
    H5Sselect_hyperslab(space_id, mode, start.data(), stride.data(), count.data(), block.data()));

  if (err < 0) {
    throw_hdf5_exception(lock, "Failed to select hyper slab");
  }
}

[[nodiscard]] legate::detail::SmallVector<hsize_t> h5s_get_simple_extents(
  const HDF5MaybeLockGuard& lock, hid_t space)
{
  const auto ndim = HDF5_CALL_NO_ERROR_PRINTING(H5Sget_simple_extent_ndims(space));

  if (ndim < 0) {
    throw_hdf5_exception(lock, "Failed to get dimensions from space");
  }

  legate::detail::SmallVector<hsize_t> ret{
    legate::detail::tags::size_tag, static_cast<std::size_t>(ndim), 0};

  const auto dims_err =
    HDF5_CALL_NO_ERROR_PRINTING(H5Sget_simple_extent_dims(space, ret.data(), nullptr));

  if (dims_err < 0) {
    throw_hdf5_exception(lock, "Failed to get dimensions from space");
  }

  return ret;
}

// ==========================================================================================

[[nodiscard]] hid_t h5t_create(const HDF5MaybeLockGuard& lock, H5T_class_t type, std::size_t size)
{
  const auto id = HDF5_CALL_NO_ERROR_PRINTING(H5Tcreate(type, size));

  if (id == H5I_INVALID_HID) {
    throw_hdf5_exception(lock,
                         fmt::format("Failed to create HDF5 type of class {} and size {}",
                                     legate::detail::to_underlying(type),
                                     size));
  }
  return id;
}

void h5t_set_tag(const HDF5MaybeLockGuard& lock, hid_t type, legate::detail::ZStringView name)
{
  const auto ret = HDF5_CALL_NO_ERROR_PRINTING(H5Tset_tag(
    type,
    name.data()  // NOLINT(bugprone-suspicious-stringview-data-usage)
    ));

  if (ret < 0) {
    throw_hdf5_exception(lock, fmt::format("Failed to set HDF5 type tag {}", name));
  }
}

namespace nothrow {

void h5t_close(hid_t hid) noexcept
{
  const auto _ = HDF5MaybeLockGuard{};

  LEGATE_CHECK(H5Tclose(hid) >= 0);
}

}  // namespace nothrow

[[nodiscard]] hid_t get_opaque_type(const HDF5MaybeLockGuard& lock, std::size_t size)
{
  class HDF5Type : public HDF5Object {
   public:
    HDF5Type() : HDF5Type{H5I_INVALID_HID} {}

    explicit HDF5Type(hid_t hid) : HDF5Object{hid, nothrow::h5t_close} {}
  };

  static auto type_cache_mutex = std::mutex{};
  static auto type_cache       = std::unordered_map<std::size_t, HDF5Type>{};

  const auto _              = std::scoped_lock{type_cache_mutex};
  const auto [it, inserted] = type_cache.try_emplace(size);

  if (inserted) {
    it->second = HDF5Type{h5t_create(lock, H5T_OPAQUE, size)};
    h5t_set_tag(lock, it->second.hid(), fmt::format("binary({})", size));
  }
  return it->second.hid();
}

// ==========================================================================================

[[nodiscard]] hid_t h5d_create2(const HDF5MaybeLockGuard& lock,
                                hid_t loc_id,
                                legate::detail::ZStringView name,
                                hid_t type,
                                hid_t space_id,
                                hid_t lcpl_id,
                                hid_t dcpl_id,
                                hid_t dapl_id)
{
  const hid_t dataset_id = HDF5_CALL_NO_ERROR_PRINTING(
    H5Dcreate2(loc_id,
               name.data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
               type,
               space_id,
               lcpl_id,
               dcpl_id,
               dapl_id));

  if (dataset_id == H5I_INVALID_HID) {
    throw_hdf5_exception(lock, fmt::format("Failed to create dataset \"{}\"", name));
  }
  return dataset_id;
}

[[nodiscard]] hid_t h5d_open(const HDF5MaybeLockGuard& lock,
                             hid_t loc_id,
                             legate::detail::ZStringView name,
                             hid_t dapl_id)
{
  const hid_t dataset_id = HDF5_CALL_NO_ERROR_PRINTING(
    H5Dopen(loc_id,
            name.data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
            dapl_id));

  if (dataset_id == H5I_INVALID_HID) {
    throw_hdf5_exception(lock, fmt::format("Failed to open dataset \"{}\"", name));
  }
  return dataset_id;
}

namespace nothrow {

void h5d_close(hid_t dset_id) noexcept
{
  const auto _ = HDF5MaybeLockGuard{};

  LEGATE_CHECK(H5Dclose(dset_id) >= 0);
}

}  // namespace nothrow

void h5d_write(const HDF5MaybeLockGuard& lock,
               hid_t dset_id,
               hid_t mem_type_id,
               hid_t mem_space_id,
               hid_t file_space_id,
               hid_t dxpl_id,
               const void* buf)
{
  const herr_t err = HDF5_CALL_NO_ERROR_PRINTING(
    H5Dwrite(dset_id, mem_type_id, mem_space_id, file_space_id, dxpl_id, buf));

  if (err < 0) {
    throw_hdf5_exception(lock, "Failed to write to disk");
  }
}

[[nodiscard]] hid_t h5d_get_type(const HDF5MaybeLockGuard& lock, hid_t dset_id)
{
  const hid_t type_id = HDF5_CALL_NO_ERROR_PRINTING(H5Dget_type(dset_id));

  if (type_id == H5I_INVALID_HID) {
    throw_hdf5_exception(lock, "Failed to get datatype of the dataset");
  }
  return type_id;
}

[[nodiscard]] hid_t h5d_get_space(const HDF5MaybeLockGuard& lock, hid_t dset_id)
{
  const hid_t space = HDF5_CALL_NO_ERROR_PRINTING(H5Dget_space(dset_id));

  if (space == H5I_INVALID_HID) {
    throw_hdf5_exception(lock, "Failed to the dataspace from the dataset");
  }

  return space;
}

// ==========================================================================================

namespace nothrow {

void h5p_close(hid_t cls_id) noexcept
{
  const auto _ = HDF5MaybeLockGuard{};

  LEGATE_CHECK(H5Pclose(cls_id) >= 0);
}

}  // namespace nothrow

[[nodiscard]] hid_t h5p_create(const HDF5MaybeLockGuard& lock, hid_t cls_id)
{
  const hid_t plist_id = HDF5_CALL_NO_ERROR_PRINTING(H5Pcreate(cls_id));

  if (plist_id == H5I_INVALID_HID) {
    throw_hdf5_exception(lock, "Failed to create property list");
  }

  return plist_id;
}

void h5p_set_virtual(const HDF5MaybeLockGuard& lock,
                     hid_t dcpl,
                     hid_t vds_space,
                     legate::detail::ZStringView file,
                     legate::detail::ZStringView src_dset,
                     hid_t src_space)
{
  const auto err = HDF5_CALL_NO_ERROR_PRINTING(
    H5Pset_virtual(dcpl,
                   vds_space,
                   // NOLINTBEGIN(bugprone-suspicious-stringview-data-usage)
                   file.data(),
                   src_dset.data(),
                   // NOLINTEND(bugprone-suspicious-stringview-data-usage)
                   src_space));

  if (err < 0) {
    throw_hdf5_exception(lock, "Failed to set virtual dataspace");
  }
}

void h5p_set_fapl_gds(const HDF5MaybeLockGuard& lock,
                      hid_t fapl_id,
                      std::size_t alignment,
                      std::size_t block_size,
                      std::size_t cbuf_size)
{
  const auto err =
    HDF5_CALL_NO_ERROR_PRINTING(H5Pset_fapl_gds(fapl_id, alignment, block_size, cbuf_size));

  if (err < 0) {
    throw_hdf5_exception(lock, "Failed to enable GDS on file access");
  }
}

}  // namespace

// ==========================================================================================

HDF5Object::HDF5Object(hid_t hid, void (*closer)(hid_t) noexcept) : hid_{hid}, closer_{closer} {}

hid_t HDF5Object::hid() const noexcept { return hid_; }

HDF5Object::HDF5Object(const HDF5Object& other)
  : hid_{h5i_inc_ref({}, other.hid())}, closer_{other.closer_}
{
}

HDF5Object& HDF5Object::operator=(const HDF5Object& other)
{
  if (this != &other) {
    // Increment refcount before calling destroy in case both objects refer to the same hid
    const auto new_ref = h5i_inc_ref({}, other.hid());

    destroy_();
    hid_    = new_ref;
    closer_ = other.closer_;
  }
  return *this;
}

HDF5Object::HDF5Object(HDF5Object&& other) noexcept
  : hid_{std::exchange(other.hid_, H5I_INVALID_HID)}, closer_{std::exchange(other.closer_, nullptr)}
{
}

HDF5Object& HDF5Object::operator=(HDF5Object&& other) noexcept
{
  if (this != &other) {
    destroy_();
    hid_    = std::exchange(other.hid_, H5I_INVALID_HID);
    closer_ = std::exchange(other.closer_, nullptr);
  }
  return *this;
}

HDF5Object::~HDF5Object() { destroy_(); }

void HDF5Object::destroy_() noexcept
{
  if (hid() != H5I_INVALID_HID) {
    closer_(hid());
    hid_    = H5I_INVALID_HID;
    closer_ = nullptr;
  }
}

// ==========================================================================================

HDF5DataSpace::HDF5DataSpace(hid_t hid) : HDF5Object{hid, nothrow::h5s_close} {}

HDF5DataSpace::HDF5DataSpace(Span<const hsize_t> sizes, Span<const hsize_t> maxdims)
  : HDF5DataSpace{h5s_create_simple({}, sizes, maxdims)}
{
}

namespace {

[[nodiscard]] H5S_seloper_t to_hdf5_seloper(const HDF5MaybeLockGuard&,
                                            HDF5DataSpace::SelectMode mode)
{
  switch (mode) {
    case HDF5DataSpace::SelectMode::SELECT_SET: return H5S_SELECT_SET;
  }
  LEGATE_ABORT("Unhandled select mode", legate::detail::to_underlying(mode));
}

}  // namespace

void HDF5DataSpace::select_hyperslab(SelectMode mode,
                                     Span<const hsize_t> start,
                                     Span<const hsize_t> stride,
                                     Span<const hsize_t> count,
                                     Span<const hsize_t> block)
{
  const auto lock = HDF5MaybeLockGuard{};

  h5s_select_hyperslab(lock, hid(), to_hdf5_seloper(lock, mode), start, stride, count, block);
}

legate::detail::SmallVector<hsize_t> HDF5DataSpace::extents() const
{
  return h5s_get_simple_extents({}, hid());
}

// ==========================================================================================

HDF5DataSet::HDF5DataSet(hid_t hid) : HDF5Object{hid, nothrow::h5d_close} {}

HDF5DataSet::HDF5DataSet(const HDF5File& file,
                         legate::detail::ZStringView name,
                         hid_t type,
                         hid_t space,
                         hid_t lcpl_id,
                         hid_t dcpl_id,
                         hid_t dapl_id)
  : HDF5DataSet{h5d_create2({}, file.hid(), name, type, space, lcpl_id, dcpl_id, dapl_id)}
{
}

namespace {

[[nodiscard]] hid_t to_hdf5_type(const HDF5MaybeLockGuard& lock, const Type& type)
{
  const auto code = type.code();
  // These macros also expand to H5open() (a library call) and therefore must be called while
  // the HDF5 mutex is held.
  switch (code) {
    case Type::Code::BOOL: return H5T_NATIVE_HBOOL;
    case Type::Code::INT8: return H5T_NATIVE_INT8;
    case Type::Code::INT16: return H5T_NATIVE_INT16;
    case Type::Code::INT32: return H5T_NATIVE_INT32;
    case Type::Code::INT64: return H5T_NATIVE_INT64;
    case Type::Code::UINT8: return H5T_NATIVE_UINT8;
    case Type::Code::UINT16: return H5T_NATIVE_UINT16;
    case Type::Code::UINT32: return H5T_NATIVE_UINT32;
    case Type::Code::UINT64: return H5T_NATIVE_UINT64;
    case Type::Code::FLOAT16: {
      // HDF5 docs note that H5T_NATIVE_FLOAT16 may be H5I_INVALID_HID if the current platform
      // doesn't support float16
      if (const auto hid = H5T_NATIVE_FLOAT16; hid != H5I_INVALID_HID) {
        return hid;
      }

#ifdef __BYTE_ORDER__
#ifdef __ORDER_LITTLE_ENDIAN__
      // NOLINTNEXTLINE(misc-redundant-expression)
      constexpr auto is_little_endian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
#elif defined(__ORDER_BIG_ENDIAN__)
      constexpr auto is_little_endian = !(__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__);
#else
#error "Cannot determine endianness of your system"
#endif
#else
      const auto is_little_endian = [] {
        constexpr std::uint16_t x = 1;

        return *reinterpret_cast<const std::uint8_t*>(&x) == 1;
      }();
#endif

      return is_little_endian ? H5T_IEEE_F16LE : H5T_IEEE_F16BE;
    }
    case Type::Code::FLOAT32: return H5T_NATIVE_FLOAT;
    case Type::Code::FLOAT64: return H5T_NATIVE_DOUBLE;
    case Type::Code::BINARY: return get_opaque_type(lock, type.size());
    case Type::Code::COMPLEX64: [[fallthrough]];
    case Type::Code::COMPLEX128: [[fallthrough]];
    case Type::Code::NIL: [[fallthrough]];
    case Type::Code::FIXED_ARRAY: [[fallthrough]];
    case Type::Code::STRUCT: [[fallthrough]];
    case Type::Code::STRING: [[fallthrough]];
    case Type::Code::LIST: break;
  }
  LEGATE_ABORT("Unhandled type code", legate::detail::to_underlying(code));
}

}  // namespace

HDF5DataSet::HDF5DataSet(const HDF5File& file,
                         legate::detail::ZStringView name,
                         const Type& type,
                         const HDF5DataSpace& space,
                         hid_t lcpl_id,
                         hid_t dcpl_id,
                         hid_t dapl_id)
  : HDF5DataSet{file,
                name,
                [&] {
                  // Need to use this temporary lambda because object lifetimes in C++ are
                  // weird. Basically, even though `to_hdf5_type()` is sequenced *before* the
                  // forwarding constructor call, the temporary lock created as part of it
                  // might have its lifetime extended to the full expression (depending on the
                  // whims of the compiler). This causes a deadlock then when the other ctor
                  // tries to re-acquire the lock to call h5d_create2().
                  //
                  // Putting it in the lambda forces the lifetime of the lock to end before
                  // the lambda returns, safely releasing the lock before the other ctor runs.
                  return to_hdf5_type({}, type);
                }(),
                space.hid(),
                lcpl_id,
                dcpl_id,
                dapl_id}
{
}

HDF5DataSet::HDF5DataSet(const HDF5File& file,
                         legate::detail::ZStringView name,
                         hid_t type,
                         const HDF5DataSpace& space,
                         hid_t lcpl_id,
                         const HDF5DataSetCreatePropertyList& dcpl,
                         hid_t dapl_id)
  : HDF5DataSet{file, name, type, space.hid(), lcpl_id, dcpl.hid(), dapl_id}
{
}

hid_t HDF5DataSet::get_type() const { return h5d_get_type({}, hid()); }

std::string HDF5DataSet::get_name() const { return h5i_get_name({}, hid()); }

HDF5DataSpace HDF5DataSet::get_data_space() const
{
  return HDF5DataSpace{h5d_get_space({}, hid())};
}

void HDF5DataSet::write(hid_t mem_space_id,
                        hid_t file_space_id,
                        hid_t dxpl_id,
                        const void* buf) const
{
  const auto ty = get_type();

  h5d_write({}, hid(), ty, mem_space_id, file_space_id, dxpl_id, buf);
}

// ==========================================================================================

namespace {

[[nodiscard]] std::uint32_t to_hdf5_open_mode(const HDF5MaybeLockGuard&, HDF5File::OpenMode mode)
{
  // The HDF5 open modes are actually macros that expand to:
  //
  // (H5check_version(...), H5open(), <open mode value>)
  //
  // So we cannot just use them directly, and need to hold our mutex. Hence this translation
  // routine.
  switch (mode) {
    case HDF5File::OpenMode::OVERWRITE: return H5F_ACC_TRUNC;
    case HDF5File::OpenMode::READ_ONLY: return H5F_ACC_RDONLY;
  }
  LEGATE_ABORT("Unhandled open mode ", legate::detail::to_underlying(mode));
}

}  // namespace

HDF5File::HDF5File(legate::detail::ZStringView filepath, OpenMode mode, hid_t fapl_id)
  : HDF5Object{[&] {
                 const auto lock    = HDF5MaybeLockGuard{};
                 const auto h5_mode = to_hdf5_open_mode(lock, mode);

                 switch (mode) {
                   case OpenMode::READ_ONLY: return h5f_open(lock, filepath, h5_mode, fapl_id);
                   case OpenMode::OVERWRITE:
                     return h5f_create(lock, filepath, h5_mode, H5P_DEFAULT, fapl_id);
                 }
                 LEGATE_ABORT("Unhandled open mode", legate::detail::to_underlying(mode));
               }(),
               nothrow::h5f_close}
{
}

HDF5File::HDF5File(legate::detail::ZStringView filepath,
                   OpenMode mode,
                   const HDF5FileAccessPropertyList& fapl)
  : HDF5File{filepath, mode, fapl.hid()}
{
}

HDF5DataSet HDF5File::get_data_set(legate::detail::ZStringView name) const
{
  return HDF5DataSet{h5d_open({}, hid(), name, H5P_DEFAULT)};
}

// ==========================================================================================

namespace {

[[nodiscard]] hid_t convert_plist_type(const HDF5MaybeLockGuard&, HDF5PropertyList::Type type)
{
  switch (type) {
    case HDF5PropertyList::Type::DATASET_CREATE: return H5P_DATASET_CREATE;
    case HDF5PropertyList::Type::FILE_ACCESS: return H5P_FILE_ACCESS;
  }
  LEGATE_ABORT("Unhandled plist type", legate::detail::to_underlying(type));
}

}  // namespace

HDF5PropertyList::HDF5PropertyList(Type type)
  : HDF5Object{[&] {
                 const auto lock          = HDF5MaybeLockGuard{};
                 const auto h5_plist_type = convert_plist_type(lock, type);

                 return h5p_create(lock, h5_plist_type);
               }(),
               nothrow::h5p_close}
{
}

// ==========================================================================================

HDF5DataSetCreatePropertyList::HDF5DataSetCreatePropertyList()
  : HDF5PropertyList{Type::DATASET_CREATE}
{
}

void HDF5DataSetCreatePropertyList::set_virtual(const HDF5DataSpace& vds_space,
                                                legate::detail::ZStringView file,
                                                const HDF5DataSet& src_dset,
                                                const HDF5DataSpace& src_space)
{
  set_virtual(vds_space, file, src_dset.get_name(), src_space);
}

void HDF5DataSetCreatePropertyList::set_virtual(const HDF5DataSpace& vds_space,
                                                legate::detail::ZStringView file,
                                                legate::detail::ZStringView src_dset_name,
                                                const HDF5DataSpace& src_space)
{
  h5p_set_virtual({}, hid(), vds_space.hid(), file, src_dset_name, src_space.hid());
}

// ==========================================================================================

HDF5FileAccessPropertyList::HDF5FileAccessPropertyList() : HDF5PropertyList{Type::FILE_ACCESS} {}

void HDF5FileAccessPropertyList::set_gds(std::size_t alignment,
                                         std::size_t block_size,
                                         std::size_t cbuf_size)
{
  h5p_set_fapl_gds({}, hid(), alignment, block_size, cbuf_size);
}

}  // namespace legate::io::hdf5::detail::wrapper
