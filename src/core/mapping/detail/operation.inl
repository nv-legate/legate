/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/mapping/detail/operation.h"

namespace legate::mapping::detail {

inline Mappable::Mappable(const Legion::Mappable* mappable)
  : Mappable{private_tag{}, MapperDataDeserializer{mappable}}
{
}

inline const mapping::detail::Machine& Mappable::machine() const { return machine_; }

inline uint32_t Mappable::sharding_id() const { return sharding_id_; }

// ==========================================================================================

inline const std::vector<InternalSharedPtr<Array>>& Task::inputs() const { return inputs_; }

inline const std::vector<InternalSharedPtr<Array>>& Task::outputs() const { return outputs_; }

inline const std::vector<InternalSharedPtr<Array>>& Task::reductions() const { return reductions_; }

inline const std::vector<Scalar>& Task::scalars() const { return scalars_; }

inline DomainPoint Task::point() const { return task_->index_point; }

// ==========================================================================================

inline const std::vector<Store>& Copy::inputs() const { return inputs_; }

inline const std::vector<Store>& Copy::outputs() const { return outputs_; }

inline const std::vector<Store>& Copy::input_indirections() const { return input_indirections_; }

inline const std::vector<Store>& Copy::output_indirections() const { return output_indirections_; }

inline DomainPoint Copy::point() const { return copy_->index_point; }

}  // namespace legate::mapping::detail
