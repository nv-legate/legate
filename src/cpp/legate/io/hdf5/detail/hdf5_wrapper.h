/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/type/types.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/zstring_view.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/span.h>

#include <H5Ipublic.h>
#include <H5Ppublic.h>
#include <H5public.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>

/**
 * @brief Return the value of a "pure" HDF5 enum, i.e. one whose value does not silently expand
 * to H5open().
 *
 * Several HDF5 enums expand to (H5open(), the_enum_value) to seamlessly initialize HDF5 before
 * the call itself. This is neat, but it messes with our global HDF5 mutex in case the
 * underlying HDF5 library wasn't built with thread-safety enabled.
 *
 * Some enums don't have this, and just expand to the value. This macro checks that these enums
 * just expand to a plain old value.
 *
 * @param __val__ The HDF5 enum value.
 *
 * @return The expanded value.
 */
#define LEGATE_PURE_H5_ENUM(__val__)                                            \
  [] {                                                                          \
    static_assert(std::string_view{LEGATE_STRINGIZE(__val__)}.find("H5open") == \
                  std::string_view::npos);                                      \
    return __val__;                                                             \
  }()

namespace legate::io::hdf5::detail::wrapper {

/**
 * @brief A global lock-guard for HDF5 in case the underlying library isn't thread-safe.
 *
 * HDF5 isn't thread-safe on all systems (it requires a specific configure-time value to be
 * enabled) so we use a global lock if it isn't. Notice, in order to avoid deadlock, we only
 * lock access to the HDF5 file API. This is because any task that blocks on the runtime will
 * get removed from the processor (still holding the mutex), then while the runtime is
 * servicing the call, another task can start running on the processor.
 */
class HDF5MaybeLockGuard {
 public:
  /**
   * @brief Construct the lock.
   *
   * If the lock needs to be taken, this will acquire and lock the mutex. Otherwise, does
   * nothing.
   */
  HDF5MaybeLockGuard();

 private:
  static inline std::mutex mutex_{};  // NOLINT(cert-err58-cp)p

  std::unique_lock<std::mutex> lock_{};
};

class HDF5File;
class HDF5PropertyList;
class HDF5DataSetCreatePropertyList;
class HDF5FileAccessPropertyList;

/**
 * @brief The base-class for wrapped HDF5 objects.
 */
class HDF5Object {
 public:
  HDF5Object() = delete;
  HDF5Object(const HDF5Object&);
  HDF5Object& operator=(const HDF5Object&);
  HDF5Object(HDF5Object&&) noexcept;
  HDF5Object& operator=(HDF5Object&&) noexcept;

  /**
   * @brief Construct a HDF5 object.
   *
   * @param hid The HDF5 identifier for the constructed object.
   * @param closer The specific closing function for the constructed object, e.g. `H5Fclose()`.
   */
  HDF5Object(hid_t hid, void (*closer)(hid_t) noexcept);

  /**
   * @return The HDF5 identifier ID for the object.
   */
  [[nodiscard]] hid_t hid() const noexcept;

  /**
   * @brief Destroys the HDF5 object if a value is contained.
   *
   * In effect, equivalent to `closer_(hid())`.
   */
  ~HDF5Object();

 private:
  /**
   * @brief Destroys the HDF5 object if a value is contained.
   *
   * In effect, equivalent to `closer_(hid())`, except that it also sets the HID to
   * H5I_INVALID_HID.
   */
  void destroy_() noexcept;

  hid_t hid_{LEGATE_PURE_H5_ENUM(H5I_INVALID_HID)};
  void (*closer_)(hid_t) noexcept {};
};

/**
 * @brief A wrapped HDF5 DataSpace object.
 */
class HDF5DataSpace : public HDF5Object {
 public:
  /**
   * @brief Construct a DataSpace from an existing HDF5 dataspace.
   *
   * This constructor does not increment the reference count for hid on construction.
   *
   * @param hid The HDF5 dataspace object.
   */
  explicit HDF5DataSpace(hid_t hid);

  /**
   * @brief Construct a new DataSpace from a pair of sizes.
   *
   * @param sizes The extents of the data space.
   * @param maxdims The maximum dimensions of the data-space, or equal to sizes if unset.
   */
  explicit HDF5DataSpace(Span<const hsize_t> sizes, Span<const hsize_t> maxdims = {});

  /**
   * @brief A wrapper over H5_seloper_t.
   */
  enum class SelectMode : std::uint8_t {
    SELECT_SET  ///< Select a subset
  };

  /**
   * @return The extents of the currently selected dataspace.
   */
  [[nodiscard]] legate::detail::SmallVector<hsize_t> extents() const;

  [[nodiscard]] std::size_t element_count() const;

  /**
   * @brief Select a subset of the DataSpace. The selection is done in place.
   *
   * @param mode The selection mode for the hyper slab.
   * @param start The position of the start of the bottom left corner of the selection.
   * @param count The extents of the selection.
   * @param stride The strides in the direction of the n'th dimension.
   * @block block The block sizes, if any.
   *
   */
  void select_hyperslab(SelectMode mode,
                        Span<const hsize_t> start,
                        Span<const hsize_t> count,
                        Span<const hsize_t> stride = {},
                        Span<const hsize_t> block  = {});
};

class HDF5Type : public HDF5Object {
 public:
  HDF5Type();
  explicit HDF5Type(hid_t hid);

  enum class Class : std::uint8_t {
    BOOL,
    SIGNED_INTEGER,
    UNSIGNED_INTEGER,
    FLOAT,
    TIME,
    STRING,
    BITFIELD,
    OPAQUE,
    COMPOUND,
    REFERENCE,
    ENUM,
    VARIABLE_LENGTH,
    ARRAY
  };

  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] Class type_class() const;
  [[nodiscard]] std::string to_string() const;
};

/**
 * @brief A wrapped HDF5 dataset object.
 */
class HDF5DataSet : public HDF5Object {
 public:
  /**
   * @brief Construct a DataSet from an existing HDF5 dataspace.
   *
   * This constructor does not increment the reference count for hid on construction.
   *
   * @param hid The HDF5 dataset object.
   */
  explicit HDF5DataSet(hid_t hid);

  /**
   * @brief Construct a new HDF5DataSet from raw HDF5 handles.
   *
   * @param file Reference to the open HDF5 file.
   * @param name Name of the dataset to create.
   * @param type HDF5 datatype identifier (hid_t).
   * @param space HDF5 dataspace identifier (hid_t).
   * @param lcpl_id Link creation property list (default: H5P_DEFAULT).
   * @param dcpl_id Dataset creation property list (default: H5P_DEFAULT).
   * @param dapl_id Dataset access property list (default: H5P_DEFAULT).
   */
  HDF5DataSet(const HDF5File& file,
              legate::detail::ZStringView name,
              hid_t type,
              hid_t space,
              hid_t lcpl_id = LEGATE_PURE_H5_ENUM(H5P_DEFAULT),
              hid_t dcpl_id = LEGATE_PURE_H5_ENUM(H5P_DEFAULT),
              hid_t dapl_id = LEGATE_PURE_H5_ENUM(H5P_DEFAULT));

  /**
   * @brief Construct a new HDF5DataSet using wrapper types.
   *
   * @param file Reference to the open HDF5 file.
   * @param name Name of the dataset to create.
   * @param type Wrapper type describing the datatype.
   * @param space HDF5DataSpace object describing the dataset shape.
   * @param lcpl_id Link creation property list (default: H5P_DEFAULT).
   * @param dcpl_id Dataset creation property list (default: H5P_DEFAULT).
   * @param dapl_id Dataset access property list (default: H5P_DEFAULT).
   */
  HDF5DataSet(const HDF5File& file,
              legate::detail::ZStringView name,
              const Type& type,
              const HDF5DataSpace& space,
              hid_t lcpl_id = LEGATE_PURE_H5_ENUM(H5P_DEFAULT),
              hid_t dcpl_id = LEGATE_PURE_H5_ENUM(H5P_DEFAULT),
              hid_t dapl_id = LEGATE_PURE_H5_ENUM(H5P_DEFAULT));

  /**
   * @brief Construct a new HDF5DataSet with explicit property list.
   *
   * @param file Reference to the open HDF5 file.
   * @param name Name of the dataset to create.
   * @param type HDF5 datatype identifier (hid_t).
   * @param space HDF5DataSpace object describing the dataset shape.
   * @param lcpl_id Link creation property list.
   * @param dcpl Dataset creation property list wrapper.
   * @param dapl_id Dataset access property list (default: H5P_DEFAULT).
   */
  HDF5DataSet(const HDF5File& file,
              legate::detail::ZStringView name,
              hid_t type,
              const HDF5DataSpace& space,
              hid_t lcpl_id,
              const HDF5DataSetCreatePropertyList& dcpl,
              hid_t dapl_id = LEGATE_PURE_H5_ENUM(H5P_DEFAULT));

  /**
   * @brief Get the HDF5 datatype identifier for this dataset.
   *
   * @return Identifier of the dataset's datatype.
   */
  [[nodiscard]] HDF5Type type() const;

  /**
   * @brief Get the name of the dataset.
   *
   * @return Dataset name.
   */
  [[nodiscard]] std::string name() const;

  /**
   * @brief Get the dataspace associated with this dataset.
   *
   * @return Dataspace object.
   */
  [[nodiscard]] HDF5DataSpace data_space() const;

  /**
   * @brief Check if the dataset is stored contiguously.
   *
   * @return True if the dataset is stored contiguously, false otherwise.
   */
  [[nodiscard]] H5D_layout_t get_layout() const;

  /**
   * @brief Get the dataset creation property list.
   *
   * @return Dataset creation property list.
   */
  [[nodiscard]] HDF5DataSetCreatePropertyList get_create_plist() const;

  /**
   * @brief Write data into the dataset.
   *
   * @param mem_space_id Identifier of the memory dataspace.
   * @param file_space_id Identifier of the file dataspace selection.
   * @param dxpl_id Data transfer property list identifier.
   * @param buf Pointer to the data buffer to write.
   */
  void write(hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void* buf) const;

  void read(hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void* buf) const;
  void read(
    hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void* buf) const;
};

/**
 * @brief A wrapped HDF5 virtual space object.
 */
class HDF5VirtualSpace : public HDF5Object {
 public:
  explicit HDF5VirtualSpace(hid_t hid, std::size_t index);

  /**
   * @brief Get the bounding box of a hyperslab selection.
   *
   * Returns the block coordinates and offset coordinates.
   *
   * @return Pair of vectors: (block coordinates, offset coordinates).
   */
  [[nodiscard]] std::pair<legate::detail::SmallVector<hsize_t>,
                          legate::detail::SmallVector<hsize_t>>
  get_select_bounds(std::size_t ndim) const;
};

/**
 * @brief Represents an HDF5 file object.
 */
class HDF5File : public HDF5Object {
 public:
  /**
   * @brief File open modes.
   */
  enum class OpenMode : std::uint8_t { OVERWRITE, READ_ONLY };

  /**
   * @brief Open or create an HDF5 file.
   *
   * @param filepath Path to the file.
   * @param mode File open mode.
   * @param fapl_id File access property list (default: H5P_DEFAULT).
   */
  HDF5File(legate::detail::ZStringView filepath,
           OpenMode mode,
           hid_t fapl_id = LEGATE_PURE_H5_ENUM(H5P_DEFAULT));

  /**
   * @brief Open or create an HDF5 file with a property list wrapper.
   *
   * @param filepath Path to the file.
   * @param mode File open mode.
   * @param fapl File access property list wrapper.
   */
  HDF5File(legate::detail::ZStringView filepath,
           OpenMode mode,
           const HDF5FileAccessPropertyList& fapl);

  /**
   * @brief Retrieve a dataset from the file.
   *
   * @param name Name of the dataset.
   *
   * @return The requested dataset.
   */
  [[nodiscard]] HDF5DataSet data_set(legate::detail::ZStringView name) const;

  [[nodiscard]] bool has_data_set(legate::detail::ZStringView dataset_name,
                                  hid_t lapl_id = LEGATE_PURE_H5_ENUM(H5P_DEFAULT)) const;
};

/**
 * @brief Represents a generic HDF5 property list.
 */
class HDF5PropertyList : public HDF5Object {
 public:
  /**
   * @brief Property list types.
   */
  enum class Type : std::uint8_t { DATASET_CREATE, FILE_ACCESS };

  /**
   * @brief Construct a property list of the given type.
   *
   * @param type The type of property list to create.
   */
  explicit HDF5PropertyList(Type type);

  /**
   * @brief Construct a property list from an existing dataset.
   *
   * @param plist_id The property list ID.
   */
  explicit HDF5PropertyList(hid_t plist_id);
};

/**
 * @brief Property list for dataset creation.
 */
class HDF5DataSetCreatePropertyList : public HDF5PropertyList {
 public:
  /**
   * @brief Construct a dataset creation property list.
   */
  HDF5DataSetCreatePropertyList();

  /**
   * @brief Construct a dataset creation property list from an existing dataset.
   *
   * @param plist_id The property list ID.
   */
  explicit HDF5DataSetCreatePropertyList(hid_t plist_id);

  /**
   * @brief Get the layout of the dataset.
   *
   * @return The layout.
   */
  [[nodiscard]] H5D_layout_t get_layout() const;

  /**
   * @brief Get the chunk dimensions for a chunked dataset.
   *
   * @param ndim Number of dimensions in the dataset.
   *
   * @return Vector of chunk dimensions, or empty if not chunked.
   */
  [[nodiscard]] legate::detail::SmallVector<hsize_t> get_chunk_dims(std::size_t ndim) const;

  /**
   * @brief Define a virtual dataset mapping to a source dataset.
   *
   * @param vds_space Dataspace of the virtual dataset.
   * @param file Source file path.
   * @param src_dset Source dataset object.
   * @param src_space Dataspace of the source dataset.
   */
  void set_virtual(const HDF5DataSpace& vds_space,
                   legate::detail::ZStringView file,
                   const HDF5DataSet& src_dset,
                   const HDF5DataSpace& src_space);

  /**
   * @brief Define a virtual dataset mapping to a source dataset by name.
   *
   * @param vds_space Dataspace of the virtual dataset.
   * @param file Source file path.
   * @param src_dset_name Name of the source dataset.
   * @param src_space Dataspace of the source dataset.
   */
  void set_virtual(const HDF5DataSpace& vds_space,
                   legate::detail::ZStringView file,
                   legate::detail::ZStringView src_dset_name,
                   const HDF5DataSpace& src_space);

  /**
   * @brief Get the number of virtual mappings in the property list.
   *
   * @return The number of virtual mappings.
   */
  [[nodiscard]] std::size_t virtual_count() const;

  /**
   * @brief Get the source filename for a virtual mapping.
   *
   * @param index Index of the virtual mapping.
   *
   * @return Source filename.
   */
  [[nodiscard]] std::string virtual_filename(std::size_t index) const;

  /**
   * @brief Get the source dataset name for a virtual mapping.
   *
   * @param index Index of the virtual mapping.
   *
   * @return Source dataset name.
   */
  [[nodiscard]] std::string virtual_dsetname(std::size_t index) const;
};

/**
 * @brief Property list for file access.
 */
class HDF5FileAccessPropertyList : public HDF5PropertyList {
 public:
  /**
   * @brief Construct a file access property list.
   */
  HDF5FileAccessPropertyList();

  /**
   * @brief Enable GPUDirect Storage (GDS) access.
   *
   * @param alignment File access alignment in bytes.
   * @param block_size Block size in bytes.
   * @param cbuf_size Chunk buffer size in bytes.
   */
  void set_gds(std::size_t alignment, std::size_t block_size, std::size_t cbuf_size);
};

}  // namespace legate::io::hdf5::detail::wrapper
