// kintera
#include "implicit.hpp"

// torch
#include <torch/torch.h>
#include <iostream>

namespace kintera {


void ImplicitSolver::explicit_step(torch::Tensor& C, torch::Tensor& rates, torch::Tensor& stoich_matrix) {
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
    C = C + dC; 
}

void ImplicitSolver::time_march(
    torch::Tensor& C,
    const torch::Tensor& P,
    const torch::Tensor& T,
    double dt,
    ReactionSystem& reaction_system) {
    
    std::cout << "checkpoint 1\n";
    // Initial guess for next timestep (using explicit step)
    auto C_next = C.clone();
    auto rates = reaction_system.calculate_rates(C, P, T);
    auto stoich_matrix = reaction_system.get_stoichiometry_matrix();
    explicit_step(C_next, rates, stoich_matrix);
    std::cout << "checkpoint 2\n";
    // Simple Newton iteration for implicit solve
    const int max_iter = 10;
    const double tol = 1e-6;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // Calculate rates at current guess
        auto rates_next = reaction_system.calculate_rates(C_next, P, T);
        std::cout << "checkpoint 3\n";
        // Calculate residual: C_next - C - dt * stoich_matrix * rates_next
        auto stoich_t = stoich_matrix.t();
        auto dC = torch::matmul(stoich_t, rates_next);
        auto residual = C_next - C - dt * dC;
        std::cout << "checkpoint 4\n";
        // Check convergence
        auto residual_norm = residual.norm().item<double>();
        if (residual_norm < tol) {
            break;
        }
        auto jacobian = torch::eye(C.size(-1), C.size(-1)) - 
                       dt * torch::matmul(stoich_t, reaction_system.calculate_jacobian(C_next, P, T));
        std::cout << "checkpoint 5\n";
        auto jacobian_inv = torch::inverse(jacobian);
        auto update = torch::matmul(jacobian_inv, residual);
        std::cout << "checkpoint 6\n";
        C_next = C_next - update;
        std::cout << "checkpoint 7\n";
        C_next.clamp_(0); // Ensure non-negativity
    }
    
    C = C_next;
}

} // namespace kintera 