#pragma once

// kintera
#include "../kinetics/reaction_system.hpp"

// torch
#include <torch/torch.h>

namespace kintera {

/**
 * @brief Base class for chemical kinetics solvers
 */
class Solver {
public:
    Solver(double max_rel_change = 0.1) : max_rel_change_(max_rel_change) {}
    virtual ~Solver() = default;

    /**
     * @brief Time march the system forward by dt
     * @param C Concentration tensor with shape (..., n_species)
     * @param P Pressure tensor with shape (..., 1)
     * @param T Temperature tensor with shape (..., 1)
     * @param dt Timestep
     * @param reaction_system The reaction system to solve
     */
    virtual void time_march(
        torch::Tensor& C,
        const torch::Tensor& P,
        const torch::Tensor& Temp,
        double dt,
        ReactionSystem& reaction_system) = 0;

protected:
    double max_rel_change_;
};

} // namespace kintera 