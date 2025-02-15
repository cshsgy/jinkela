#pragma once

#include <vector>
#include <string>
#include <torch/torch.h>
#include "../reaction.hpp"

namespace kintera {

// The matrix has dimensions (number of reactions × number of species).
torch::Tensor generate_stoichiometry_matrix(
    const std::vector<Reaction>& reactions,
    const std::vector<std::string>& species);

} // namespace kintera
