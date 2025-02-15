// C/C++
#include <algorithm>
#include <stdexcept>

// kintera
#include "stoichiometry.hpp"

namespace kintera {

torch::Tensor generate_stoichiometry_matrix(
    const std::map<Reaction, torch::nn::AnyModule>& reactions,
    const std::vector<std::string>& species) {
  auto matrix = torch::zeros(
      {static_cast<long>(reactions.size()), static_cast<long>(species.size())});

  for (auto it = reactions.begin(); it != reactions.end(); ++it) {
    size_t i = std::distance(reactions.begin(), it);
    for (const auto& [species_name, coeff] : it->first.reactants()) {
      auto jt = std::find(species.begin(), species.end(), species_name);
      if (jt == species.end()) {
        throw std::runtime_error("Species " + species_name +
                                 " not found in species list");
      }
      size_t j = std::distance(species.begin(), jt);
      matrix[i][j] -= coeff;
    }

    for (const auto& [species_name, coeff] : it->first.products()) {
      auto jt = std::find(species.begin(), species.end(), species_name);
      if (jt == species.end()) {
        throw std::runtime_error("Species " + species_name +
                                 " not found in species list");
      }
      size_t j = std::distance(species.begin(), jt);
      matrix[i][j] += coeff;
    }
  }

  return matrix;
}

}  // namespace kintera
