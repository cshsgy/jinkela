#pragma once

// kintera
#include "solver.hpp"

// torch
#include <torch/torch.h>

namespace kintera {

/**
 * @brief Explicit solver for chemical kinetics
 */
class ExplicitSolver : public Solver {
public:
    using Solver::Solver;

    void time_march(
        torch::Tensor& C,
        const torch::Tensor& P,
        const torch::Tensor& Temp,
        double dt,
        ReactionSystem& reaction_system) override;
};

} // namespace kintera
