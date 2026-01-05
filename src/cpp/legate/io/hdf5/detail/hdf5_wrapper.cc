/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <algorithm>
#include <array>
#include <cctype>
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
}

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

[[nodiscard]] std::size_t h5s_get_simple_extents_npoints(const HDF5MaybeLockGuard& lock,
                                                         hid_t space)
{
  const auto ret = HDF5_CALL_NO_ERROR_PRINTING(H5Sget_simple_extent_npoints(space));

  if (ret < 0) {
    throw_hdf5_exception(lock, "Failed to get number of points from space");
  }
  return static_cast<std::size_t>(ret);
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

[[nodiscard]] hid_t h5t_enum_create(const HDF5MaybeLockGuard& lock, hid_t base_id)
{
  const auto id = HDF5_CALL_NO_ERROR_PRINTING(H5Tenum_create(base_id));

  if (id == H5I_INVALID_HID) {
    throw_hdf5_exception(lock, "Failed to create HDF5 enum type");
  }
  return id;
}

void h5t_enum_insert(const HDF5MaybeLockGuard& lock,
                     hid_t enum_id,
                     legate::detail::ZStringView name,
                     const void* value)
{
  const auto err = HDF5_CALL_NO_ERROR_PRINTING(
    H5Tenum_insert(enum_id,
                   name.data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
                   value));

  if (err < 0) {
    throw_hdf5_exception(lock, fmt::format("Failed to insert enum value {}={}", name, value));
  }
}

[[nodiscard]] std::uint32_t h5t_get_nmembers(const HDF5MaybeLockGuard& lock, hid_t type_id)
{
  const auto ret = HDF5_CALL_NO_ERROR_PRINTING(H5Tget_nmembers(type_id));

  if (ret < 0) {
    throw_hdf5_exception(lock, "Failed to get HDF5 enum number of members");
  }
  return static_cast<std::uint32_t>(ret);
}

[[nodiscard]] std::string h5t_get_member_name(const HDF5MaybeLockGuard& lock,
                                              hid_t type_id,
                                              std::uint32_t mem_idx)
{
  auto* const name = HDF5_CALL_NO_ERROR_PRINTING(H5Tget_member_name(type_id, mem_idx));

  if (name == nullptr) {
    throw_hdf5_exception(lock, fmt::format("Failed to get HDF5 type member name {}", mem_idx));
  }

  auto ret = [&] {
    try {
      return std::string{name};
    } catch (...) {
      LEGATE_CHECK(H5free_memory(name) >= 0);
      throw;
    }
  }();

  const auto err = HDF5_CALL_NO_ERROR_PRINTING(H5free_memory(name));

  if (err < 0) {
    throw_hdf5_exception(lock, fmt::format("Failed to free HDF5 type member name {}", ret));
  }
  return ret;
}

[[nodiscard]] std::size_t h5t_get_size(const HDF5MaybeLockGuard& lock, hid_t type_id)
{
  const auto ret = HDF5_CALL_NO_ERROR_PRINTING(H5Tget_size(type_id));

  if (ret == 0) {
    throw_hdf5_exception(lock, "Failed to get HDF5 type size");
  }
  return ret;
}

[[nodiscard]] H5T_class_t h5t_get_class(const HDF5MaybeLockGuard& lock, hid_t type_id)
{
  const auto ret = HDF5_CALL_NO_ERROR_PRINTING(H5Tget_class(type_id));

  if (ret == H5T_NO_CLASS) {
    throw_hdf5_exception(lock, "Failed to get HDF5 type class");
  }
  return ret;
}

[[nodiscard]] H5T_sign_t h5t_get_sign(const HDF5MaybeLockGuard& lock, hid_t type_id)
{
  const auto ret = HDF5_CALL_NO_ERROR_PRINTING(H5Tget_sign(type_id));

  if (ret == H5T_SGN_ERROR) {
    throw_hdf5_exception(lock, "Failed to get HDF5 type sign");
  }
  return ret;
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

void h5d_read(const HDF5MaybeLockGuard& lock,
              hid_t dset_id,
              hid_t mem_type_id,
              hid_t mem_space_id,
              hid_t file_space_id,
              hid_t dxpl_id,
              void* buf)
{
  const herr_t err = HDF5_CALL_NO_ERROR_PRINTING(
    H5Dread(dset_id, mem_type_id, mem_space_id, file_space_id, dxpl_id, buf));

  if (err < 0) {
    throw_hdf5_exception(lock, "Failed to read from disk");
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

/**
 * @brief Get the creation property list from a dataset.
 *
 * @param lock The lock to use.
 * @param dset_id The dataset ID.
 * @return The creation property list from the dataset.
 */
[[nodiscard]] hid_t h5d_get_create_plist(const HDF5MaybeLockGuard& lock, hid_t dset_id)
{
  const hid_t plist_id = HDF5_CALL_NO_ERROR_PRINTING(H5Dget_create_plist(dset_id));

  if (plist_id == H5I_INVALID_HID) {
    throw_hdf5_exception(lock, "Failed to get creation property list from the dataset");
  }

  return plist_id;
}

/**
 * @brief Get the layout of a property list.
 *
 * @param lock The lock to use.
 * @param plist_id The property list ID.
 * @return The layout of the property list.
 */
[[nodiscard]] H5D_layout_t h5p_get_layout(const HDF5MaybeLockGuard& lock, hid_t plist_id)
{
  const H5D_layout_t layout = HDF5_CALL_NO_ERROR_PRINTING(H5Pget_layout(plist_id));

  if (layout == H5D_LAYOUT_ERROR) {
    throw_hdf5_exception(lock, "Failed to get layout from property list");
  }

  return layout;
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

// ==========================================================================================

[[nodiscard]] bool h5l_exists(const HDF5MaybeLockGuard& lock,
                              hid_t loc_id,
                              legate::detail::ZStringView name,
                              hid_t lapl_id)
{
  // The root path always exists, but H5Lexists return 0 or 1 depending of the version of HDF5,
  // so always return true for it.
  if (name == "/") {
    return true;
  }

  const auto ret = HDF5_CALL_NO_ERROR_PRINTING(
    H5Lexists(loc_id,
              name.data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
              lapl_id));

  if (ret < 0) {
    throw_hdf5_exception(lock, "Invalid link for exist()");
  }
  return ret > 0;
}

// ==========================================================================================

[[nodiscard]] H5O_info_t h5o_get_info_by_name(const HDF5MaybeLockGuard& lock,
                                              hid_t loc_id,
                                              legate::detail::ZStringView name,
                                              std::uint32_t fields,
                                              hid_t lapl_id)
{
  H5O_info_t ret{};

  const auto err = HDF5_CALL_NO_ERROR_PRINTING(
    H5Oget_info_by_name(loc_id,
                        name.data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
                        &ret,
                        fields,
                        lapl_id));

  if (err < 0) {
    throw_hdf5_exception(lock, "Invalid object for info");
  }
  return ret;
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
                                     Span<const hsize_t> count,
                                     Span<const hsize_t> stride,
                                     Span<const hsize_t> block)
{
  const auto lock = HDF5MaybeLockGuard{};

  h5s_select_hyperslab(lock, hid(), to_hdf5_seloper(lock, mode), start, stride, count, block);
}

legate::detail::SmallVector<hsize_t> HDF5DataSpace::extents() const
{
  return h5s_get_simple_extents({}, hid());
}

std::size_t HDF5DataSpace::element_count() const
{
  return h5s_get_simple_extents_npoints({}, hid());
}

// ==========================================================================================

HDF5Type::HDF5Type() : HDF5Type{H5I_INVALID_HID} {}

HDF5Type::HDF5Type(hid_t hid) : HDF5Object{hid, nothrow::h5t_close} {}

std::size_t HDF5Type::size() const { return h5t_get_size({}, hid()); }

namespace {

[[nodiscard]] std::string string_tolower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

}  // namespace

HDF5Type::Class HDF5Type::type_class() const
{
  const auto lock = HDF5MaybeLockGuard{};
  const auto cls  = h5t_get_class(lock, hid());

  switch (cls) {
    case H5T_INTEGER: {
      const auto sign = h5t_get_sign(lock, hid());

      return sign == H5T_SGN_NONE ? Class::UNSIGNED_INTEGER : Class::SIGNED_INTEGER;
    }
    case H5T_FLOAT: return Class::FLOAT;
    case H5T_TIME: return Class::TIME;
    case H5T_STRING: return Class::STRING;
    case H5T_BITFIELD: return Class::BITFIELD;
    case H5T_OPAQUE: return Class::OPAQUE;
    case H5T_COMPOUND: return Class::COMPOUND;
    case H5T_REFERENCE: return Class::REFERENCE;
    case H5T_ENUM: {
      if (const auto nmembers = h5t_get_nmembers(lock, hid()); nmembers == 2) {
        const auto name1          = string_tolower(h5t_get_member_name(lock, hid(), /*mem_idx=*/0));
        const auto name2          = string_tolower(h5t_get_member_name(lock, hid(), /*mem_idx=*/1));
        constexpr auto is_boolean = [](std::string_view name) {
          return name == "true" || name == "false";
        };

        if (is_boolean(name1) && is_boolean(name2)) {
          return Class::BOOL;
        }
      }
      return Class::ENUM;
    }
    case H5T_VLEN: return Class::VARIABLE_LENGTH;
    case H5T_ARRAY: return Class::ARRAY;
    case H5T_NO_CLASS: [[fallthrough]];
    case H5T_NCLASSES: break;
  }
  LEGATE_ABORT("Unhandled class type", legate::detail::to_underlying(cls));
}

std::string HDF5Type::to_string() const
{
  const auto type_class_to_string = [=]() -> std::string_view {
    switch (type_class()) {
      case Class::BOOL: return "bool";
      case Class::SIGNED_INTEGER: return "int";
      case Class::UNSIGNED_INTEGER: return "uint";
      case Class::FLOAT: return "float";
      case Class::TIME: return "time";
      case Class::STRING: return "string";
      case Class::BITFIELD: return "bitfield";
      case Class::OPAQUE: return "opaque";
      case Class::COMPOUND: return "compound";
      case Class::REFERENCE: return "reference";
      case Class::ENUM: return "enum";
      case Class::VARIABLE_LENGTH: return "variable_length";
      case Class::ARRAY: return "array";
    }
    // Unreachable
    LEGATE_ABORT("Unhandled type class");
  }();

  return fmt::format("{}({})", type_class_to_string, size());
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

[[nodiscard]] HDF5Type make_bool_type(const HDF5MaybeLockGuard& lock)
{
  auto ret                 = HDF5Type{h5t_enum_create(lock, H5T_NATIVE_HBOOL)};
  constexpr auto false_val = false;
  constexpr auto true_val  = true;

  h5t_enum_insert(lock, ret.hid(), "FALSE", &false_val);
  h5t_enum_insert(lock, ret.hid(), "TRUE", &true_val);
  return ret;
}

[[nodiscard]] hid_t to_hdf5_type(const HDF5MaybeLockGuard& lock, const Type& type)
{
  const auto code = type.code();
  // These macros also expand to H5open() (a library call) and therefore must be called while
  // the HDF5 mutex is held.
  switch (code) {
    case Type::Code::BOOL: {
      // We cannot use H5T_NATIVE_HBOOL directly, because it's not a "real" boolean
      // type. Instead, it's an alias to whichever host type is closest, usually uint8_t. This
      // makes it indistinguishuable from a truex std::uint8_t when reading again.
      //
      // So make a custom enum with true/false values in the hopes that other readers can use
      // it properly.
      static const auto bool_type = make_bool_type(lock);

      return bool_type.hid();
    }
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

HDF5Type HDF5DataSet::type() const { return HDF5Type{h5d_get_type({}, hid())}; }

std::string HDF5DataSet::name() const { return h5i_get_name({}, hid()); }

HDF5DataSpace HDF5DataSet::data_space() const { return HDF5DataSpace{h5d_get_space({}, hid())}; }

H5D_layout_t HDF5DataSet::get_layout() const
{
  return HDF5DataSetCreatePropertyList{hid()}.get_layout();
}

void HDF5DataSet::write(hid_t mem_space_id,
                        hid_t file_space_id,
                        hid_t dxpl_id,
                        const void* buf) const
{
  const auto ty = type();

  h5d_write({}, hid(), ty.hid(), mem_space_id, file_space_id, dxpl_id, buf);
}

void HDF5DataSet::read(hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void* buf) const
{
  read(type().hid(), mem_space_id, file_space_id, dxpl_id, buf);
}

void HDF5DataSet::read(
  hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void* buf) const
{
  h5d_read({}, hid(), mem_type_id, mem_space_id, file_space_id, dxpl_id, buf);
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

HDF5DataSet HDF5File::data_set(legate::detail::ZStringView name) const
{
  return HDF5DataSet{h5d_open({}, hid(), name, H5P_DEFAULT)};
}

bool HDF5File::has_data_set(legate::detail::ZStringView dataset_name, hid_t lapl_id) const
{
  const auto lock = HDF5MaybeLockGuard{};

  if (!h5l_exists(lock, hid(), dataset_name, lapl_id)) {
    return false;
  }

  const auto info = h5o_get_info_by_name(lock, hid(), dataset_name, H5O_INFO_BASIC, lapl_id);

  return info.type == H5O_TYPE_DATASET;
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

HDF5PropertyList::HDF5PropertyList(hid_t plist_id) : HDF5Object{plist_id, nothrow::h5p_close} {}

// ==========================================================================================

HDF5DataSetCreatePropertyList::HDF5DataSetCreatePropertyList()
  : HDF5PropertyList{Type::DATASET_CREATE}
{
}

HDF5DataSetCreatePropertyList::HDF5DataSetCreatePropertyList(hid_t plist_id)
  : HDF5PropertyList{[&] {
      const auto lock = HDF5MaybeLockGuard{};
      return h5d_get_create_plist(lock, plist_id);
    }()}
{
}

H5D_layout_t HDF5DataSetCreatePropertyList::get_layout() const { return h5p_get_layout({}, hid()); }

void HDF5DataSetCreatePropertyList::set_virtual(const HDF5DataSpace& vds_space,
                                                legate::detail::ZStringView file,
                                                const HDF5DataSet& src_dset,
                                                const HDF5DataSpace& src_space)
{
  set_virtual(vds_space, file, src_dset.name(), src_space);
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
