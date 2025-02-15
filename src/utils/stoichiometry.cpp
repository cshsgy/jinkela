#include "stoichiometry.hpp"
#include <algorithm>
#include <stdexcept>

namespace kintera {

torch::Tensor generate_stoichiometry_matrix(
    const std::vector<Reaction>& reactions,
    const std::vector<std::string>& species) {
    
    auto matrix = torch::zeros({static_cast<long>(reactions.size()),
                              static_cast<long>(species.size())});
    
    for (size_t i = 0; i < reactions.size(); ++i) {
        const auto& reaction = reactions[i];
        for (const auto& [species_name, coeff] : reaction.reactants()) {
            auto it = std::find(species.begin(), species.end(), species_name);
            if (it == species.end()) {
                throw std::runtime_error("Species " + species_name + " not found in species list");
            }
            size_t j = std::distance(species.begin(), it);
            matrix[i][j] -= coeff;
        }
        
        for (const auto& [species_name, coeff] : reaction.products()) {
            auto it = std::find(species.begin(), species.end(), species_name);
            if (it == species.end()) {
                throw std::runtime_error("Species " + species_name + " not found in species list");
            }
            size_t j = std::distance(species.begin(), it);
            matrix[i][j] += coeff;
        }
    }
    
    return matrix;
}

} // namespace kintera
