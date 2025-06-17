#pragma once

// kintera
#include "solver.hpp"

// torch
#include <torch/torch.h>

namespace kintera {

/**
 * @brief Implicit solver for chemical kinetics using backward Euler method
 */
class ImplicitSolver : public Solver {
public:
    using Solver::Solver;

    void time_march(
        torch::Tensor& C,
        const torch::Tensor& P,
        const torch::Tensor& Temp,
        double dt,
        ReactionSystem& reaction_system) override;

private:
    void explicit_step(torch::Tensor& C, torch::Tensor& rates, torch::Tensor& stoich_matrix);
};

} // namespace kintera 