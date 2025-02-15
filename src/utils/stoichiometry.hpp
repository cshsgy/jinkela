#pragma once

// C/C++
#include <string>
#include <vector>

// torch
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/reaction.hpp>

namespace kintera {

// The matrix has dimensions (number of reactions × number of species).
torch::Tensor generate_stoichiometry_matrix(
    const std::map<Reaction, torch::nn::AnyModule>& reactions,
    const std::vector<std::string>& species);

}  // namespace kintera
