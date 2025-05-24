// kintera
#include "../kinetics/reaction_system.hpp"
#include "explicit.hpp"

// torch
#include <torch/torch.h>
#include <iostream>

namespace kintera {

void ExplicitSolver::time_march(
    torch::Tensor& C,
    const torch::Tensor& P,
    const torch::Tensor& Temp,
    double dt,
    ReactionSystem& reaction_system) {
    
    auto rates = reaction_system.calculate_rates(C, P, Temp);
    auto stoich_matrix = reaction_system.get_stoichiometry_matrix();
    
    auto stoich_t = stoich_matrix.t();
    
    std::vector<int64_t> expand_shape;
    for (int64_t i = 0; i < rates.dim() - 1; ++i) {
        expand_shape.push_back(rates.size(i));
    }
    expand_shape.push_back(stoich_t.size(0));
    expand_shape.push_back(stoich_t.size(1));
    
    auto stoich_expanded = stoich_t.expand(expand_shape);
    auto rates_reshaped = rates.unsqueeze(-1);
    auto dC = torch::matmul(stoich_expanded, rates_reshaped).squeeze(-1);
    auto abs_rel_changes = (dC * dt).abs() / (C + 1e-30);
    auto max_change = abs_rel_changes.max().template item<double>();
    
    // Limit timestep if necessary
    double actual_dt = dt;
    if (max_change > this->max_rel_change_) {
        std::cout << "Limiting timestep from " << dt << " to " 
                 << dt * (this->max_rel_change_ / max_change) << std::endl;
        actual_dt = dt * (this->max_rel_change_ / max_change);
    }
    
    C += dC * actual_dt;
    C.clamp_(0);  // Ensure non-negativity
}

} // namespace kintera
