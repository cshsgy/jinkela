#include "kinetics.h"
#include "../utils/stoichiometry.hpp"
#include <algorithm>
#include <iostream>

// TODO: cuda kernel acceleration with a matrix-parsed input
namespace kintera {

void calculate_reaction_rates(
    const torch::Tensor& T,
    const torch::Tensor& P,
    const torch::Tensor& C,
    torch::Tensor &rates,
    const std::vector<Reaction>& reactions,
    const std::vector<std::string>& species_names) {
    
    auto batch_dims = T.sizes().vec();
    const int n_reactions = reactions.size();
    
    // Calculate rate for each reaction
    for (int i = 0; i < n_reactions; ++i) {
        const auto& rxn = reactions[i];
        auto rateCoeff = rxn.rate->evalRate(T, P);
                
        // Calculate concentration product term
        auto conc_prod = torch::ones_like(T);
        for (const auto& [species, order] : rxn.orders()) {
            auto species_idx = std::find(species_names.begin(), species_names.end(), species) - species_names.begin();
            conc_prod *= torch::pow(C.select(-1, species_idx), order);
        }
        rates.select(-1, i) = rateCoeff * conc_prod;
    }
}

void calculate_jacobian(
    const torch::Tensor& T,
    const torch::Tensor& P,
    const torch::Tensor& C,
    torch::Tensor &jacobian,
    const std::vector<Reaction>& reactions,
    const std::vector<std::string>& species_names) {
    
    auto batch_dims = C.sizes().vec();
    const int n_species = batch_dims.back();
    batch_dims.pop_back();
    const int n_reactions = reactions.size();
    
    for (int i = 0; i < n_reactions; ++i) {
        const auto& rxn = reactions[i];
        auto base_rate = rxn.rate->evalRate(T, P);
        
        for (int j = 0; j < n_species; ++j) {
            const std::string& species = species_names[j];
            
            // Skip if species is not involved in reaction
            auto order_it = rxn.orders().find(species);
            if (order_it == rxn.orders().end()) {
                continue;
            }
            
            auto conc_prod = torch::ones_like(T);
            for (const auto& [sp, ord] : rxn.orders()) {
                if (sp != species) {
                    auto sp_idx = std::find(species_names.begin(), species_names.end(), sp) - species_names.begin();
                    conc_prod *= torch::pow(C.select(-1, sp_idx), ord);
                }
            }
            
            double order = order_it->second;
            auto species_conc = C.select(-1, j);
            auto partial = base_rate * conc_prod * order * torch::pow(species_conc, order - 1);
            
            jacobian.select(-2, i).select(-1, j) = partial;
        }
        
    }
}

} // namespace kintera 