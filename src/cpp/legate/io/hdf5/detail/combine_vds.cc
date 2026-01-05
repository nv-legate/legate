/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/combine_vds.h>

#include <legate/io/hdf5/detail/hdf5_wrapper.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/small_vector.h>

#include <fmt/format.h>
#include <fmt/std.h>

#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
// GCC 14 alloc-zero warning when using Conda installed compiler
//
// In file included from <libstdc++>/regex:69,
//                  from <project>/combine_vds.cc:20:
//
// In constructor 'std::__detail::_Executor<...>::_State_info<_ResultsVec>::_State_info(...)',
//     inlined from 'std::__detail::_Executor<...>::_Executor(...)' at
// <libstdc++>/bits/regex_executor.h:81:2,
//     inlined from 'bool std::__detail::_Executor<...>::_M_lookahead(...)' at
//     <libstdc++>/bits/regex_executor.tcc:155:17:
// <libstdc++>/bits/regex_executor.h:244:31: error: argument 1 value is zero
// [-Werror=alloc-zero]
//   244 |           : _M_visited_states(new bool[__n]()), _M_start(__start)
//       |                                        ^^
//
// Not much we can do but silence it.
LEGATE_PRAGMA_PUSH();
LEGATE_PRAGMA_GCC_IGNORE("-Walloc-zero");
#include <regex>
LEGATE_PRAGMA_POP();
#include <string>
#include <string_view>
#include <system_error>

namespace legate::io::hdf5::detail {

namespace {

/**
 * @brief Parse a domain point from an encoded string.
 *
 * This function assumes that `sv` contains a `_`-separated string of digits:
 * ```
 * [0-9]+_[0-9]+_[0-9]+_...
 * ```
 *
 * Where each digit `d_i` is the `i`th entry in `DomainPoint::point_data`. So a 3 dimensional
 * `DomainPoint(1,2,3)` would be encoded as `1_2_3`.
 *
 * @params sv The string to parse from.
 *
 * @return The DomainPoint.
 */
[[nodiscard]] DomainPoint parse_to_domain_point(std::string_view sv)
{
  DomainPoint ret{};
  std::uint32_t dim = 0;

  while (!sv.empty()) {
    const auto off = sv.find('_');
    const auto sub = sv.substr(0, off);
    auto coord     = coord_t{};

    if (const auto [_, ec] = std::from_chars(sub.begin(), sub.end(), coord); ec != std::errc{}) {
      if (ec == std::errc::invalid_argument) {
        LEGATE_ABORT(fmt::format(
          "{} is not a valid value. Expected an integral or floating point value", sub));
      }
      if (ec == std::errc::result_out_of_range) {
        LEGATE_ABORT(fmt::format("{} is not a valid value, must be in [{}, {}]",
                                 sub,
                                 std::numeric_limits<coord_t>::min(),
                                 std::numeric_limits<coord_t>::max()));
      }
      LEGATE_ABORT(
        fmt::format("Unknown error parsing {}, found {} ({})", sv, sub, std::make_error_code(ec)));
    }

    ret[dim++] = coord;

    sv.remove_prefix(sub.size());
    if (!sv.empty() && sv.front() == '_') {
      sv.remove_prefix(1);
    }
  }
  ret.dim = static_cast<std::int32_t>(dim);
  return ret;
}

/**
 * @brief Given the bounds.txt file written by the HDF5WriteVDS task, parse the upper and lower
 * bounds of the tasks.
 *
 * `bounds.txt` is assumed to be in the following format:
 * ```
 * [0-9]+_[0-9]+_...
 * [0-9]+_[0-9]+_...
 * ```
 *
 * The first entry is `launch_domain().lo()` as a `_`-separated string, the second entry is
 * `launch_domain().hi()` in the same format. For example, a 2D launch domain of `(<0, 0, 0>,
 * <4, 5, 6>)` would leave behind a `bounds.txt` containing exactly:
 * ```
 * 0_0_0
 * 4_5_6
 * ```
 *
 * @param bounds_txt The `bounds.txt` file.
 *
 * @return The launch domain of the parallel write task.
 */
[[nodiscard]] Domain read_bounds(const std::filesystem::path& bounds_txt)
{
  std::string lo_line;
  std::string hi_line;

  {
    auto f = std::ifstream{bounds_txt};

    LEGATE_CHECK(std::getline(f, lo_line).good());
    LEGATE_CHECK(std::getline(f, hi_line).good());
  }
  return {parse_to_domain_point(lo_line), parse_to_domain_point(hi_line)};
}

/**
 * @brief For all intents and purposes, a Domain, except that it just holds lo and hi directly,
 * because Domain returns these by copy (as they are stored in interleaved form).
 *
 * Since we parse lo and hi separately from the file it makes no sense to needlessly tangle
 * then untangle them.
 */
class UnwrappedDomain {
 public:
  DomainPoint lo{};
  DomainPoint hi{};
};

/**
 * @brief Parse relevant information from a potential partial VDS file.
 *
 * @param path The full path to a file potentially written by the HDF5WriteVDS task.
 *
 * @return The domain of the sub-tile if `path` is a valid sub-file, `std::nullopt` otherwise.
 */
[[nodiscard]] std::optional<UnwrappedDomain> parse_sub_vds_file(const std::filesystem::path& path,
                                                                const Domain& launch_domain)
{
  static const auto sub_file_regex =
    std::regex{R"(([\d_]+)_lo_([\d_]+)_hi_([\d_]+))", std::regex::optimize};

  // Need to copy here in order to keep stem alive (which returns by value anyway) for smatch
  // etc.
  const auto stem = path.stem();
  auto cm         = std::smatch{};

  if (!std::regex_match(stem.native(), cm, sub_file_regex)) {
    return std::nullopt;
  }

  const auto get_domain_point = [](const std::ssub_match& sm) {
    const auto sv = std::string_view{sm.first.base(),
                                     static_cast<std::size_t>(std::distance(sm.first, sm.second))};

    LEGATE_CPP_VERSION_TODO(20, "Use string_view begin-end ctor above");
    return parse_to_domain_point(sv);
  };

  if (const auto index_point = get_domain_point(cm[1]); !launch_domain.contains(index_point)) {
    // A sub vds file left behind by a previous iteration of this task
    return std::nullopt;
  }

  auto ret = UnwrappedDomain{/* lo */ get_domain_point(cm[2]),
                             /* hi */ get_domain_point(cm[3])};

  LEGATE_CHECK(ret.lo.get_dim() == ret.hi.get_dim());
  return ret;
}

}  // namespace

/*static*/ void HDF5CombineVDS::cpu_variant(legate::TaskContext context)
{
  const auto vds_file     = context.scalar(0).value<std::string>();
  const auto vds_dir      = std::filesystem::path{context.scalar(1).value<std::string_view>()};
  const auto global_shape = context.scalar(2).values<std::uint64_t>();
  const auto dset_name    = context.scalar(3).value<std::string>();
  const auto domain       = read_bounds(vds_dir / "bounds.txt");

  auto vds_space = wrapper::HDF5DataSpace{global_shape};
  auto vds_plist = wrapper::HDF5DataSetCreatePropertyList{};
  auto start =
    legate::detail::SmallVector<hsize_t>{legate::detail::tags::size_tag, global_shape.size(), 0};
  auto extents   = start;
  auto src_dtype = std::optional<wrapper::HDF5Type>{};

  // HDF5WriteVDS leaf tasks each write their local section of the array to a separate file
  // under `vds_dir`. The final step of VDS construction creates a super-file that stitches
  // these sub-files together and records a mapping from global array indices to the actual
  // data locations in the sub-files.
  //
  // To build this mapping, several pieces of information must be reconstructed from the
  // sub-files:
  //
  // - Their extents, which may not be uniform, since different leaf tasks may receive
  //   differently sized chunks of the original array.
  // - Their offsets, which describe where the sub-arrays belong in the global array. This
  //   allows us to answer: given element (i,j,k), which sub-file contains it and where in that
  //   file?
  // - The data type (which is the same across all sub-arrays.)
  //
  // To simplify reconstruction and improve performance, most of the required metadata is
  // embedded in the sub-file name. The format is:
  //
  // ```
  // [0-9_]+_lo_[0-9_]+_hi_[0-9_]+
  // ```
  //
  // Each [0-9_]+ is a domain point encoded according to parse_to_domain_point() above. The
  // first domain point is the leaf-task index point, the second is the lo domain point of the
  // local array, and the third is the hi domain point of the local array.
  //
  // The leaf task index point encoded in each sub-file name is what lets us decide if that
  // file should be included in the current stitching. When the same output file name is reused
  // by the user, sub-files from earlier runs may still be sitting in `vds_dir`. To prevent
  // including stale files, the leaf tasks write an additional file, bounds.txt, alongside the
  // sub-files. This file contains the `lo` and `hi` index points of the launch domain that
  // define the full range of sub-files written by the most recent HDF5WriteTask.
  //
  // Since every leaf task writes exactly one sub-file for each index point in that range, any
  // sub-file whose index point lies within [lo, hi] is guaranteed to belong to the current
  // run. Conversely, any sub-file outside that range must be a leftover from a previous run
  // and is ignored.
  //
  // --------------------
  //
  // The lo and hi are used to compute the start offsets and extents of the sub-rectangle of
  // contained by that sub-file.
  for (auto&& entry : std::filesystem::directory_iterator{
         vds_dir, std::filesystem::directory_options::skip_permission_denied}) {
    if (!entry.is_regular_file()) {
      // Somehow a directory, or other file crept into our "private" vds directory. In any
      // case, it's not one of ours, so ignore it.
      continue;
    }

    const auto sub_domain = parse_sub_vds_file(entry, domain);

    if (!sub_domain.has_value()) {
      // Not one of the sub vds files corresponding to this stitching
      continue;
    }

    auto&& path = entry.path().native();

    if (!src_dtype.has_value()) {
      // All the sub-files should contain data of the same data-type, so we only need to read
      // the type out from one of them once. We could try and do this ourselves, but then we
      // would need to pass the full Type object down to this task, which is presently not
      // possible.
      //
      // Passing the full store (just to get the type) is also not good. This task is always
      // run as a singleton task, and passing the store as input would make Legion think we
      // mean to actually access the data. Thus the entire store would be gathered to a single
      // core, on a single node just so we can see its type. It's easier (and arguably less
      // error-prone) to just read it from disk.
      const auto file = wrapper::HDF5File{path, wrapper::HDF5File::OpenMode::READ_ONLY};

      src_dtype = file.data_set(dset_name).type();
    }

    LEGATE_CHECK(sub_domain->lo.get_dim() == static_cast<int>(start.size()));
    LEGATE_CHECK(sub_domain->lo.get_dim() == static_cast<int>(extents.size()));

    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(sub_domain->lo.get_dim()); ++i) {
      start[i] = static_cast<hsize_t>(sub_domain->lo[i]);
      extents[i] =
        static_cast<hsize_t>(std::max(sub_domain->hi[i] - sub_domain->lo[i] + 1, coord_t{0}));
    }

    vds_space.select_hyperslab(wrapper::HDF5DataSpace::SelectMode::SELECT_SET, start, extents);
    vds_plist.set_virtual(vds_space, path, dset_name, wrapper::HDF5DataSpace{extents});
  }

  LEGATE_CHECK(src_dtype.has_value());

  // No need to call stitched_dset.write() (we have no new data to write anyways), the
  // destructors of these objects ensure that HDF5 creates the stitched file.
  const auto stitched_file = wrapper::HDF5File{vds_file, wrapper::HDF5File::OpenMode::OVERWRITE};
  const auto stitched_dset = wrapper::HDF5DataSet{
    stitched_file, dset_name, src_dtype->hid(), vds_space, H5P_DEFAULT, vds_plist, H5P_DEFAULT};
}

}  // namespace legate::io::hdf5::detail
