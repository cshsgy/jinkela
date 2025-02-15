#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

#include "../reaction.hpp"

namespace kintera {
/**
 * @brief Calculate reaction rates for a set of reactions
 * @param T Temperature tensor [K]
 * @param P Pressure tensor [Pa]
 * @param C Concentration tensor [kmol/m³] with shape (..., N_species)
 * @param reactions List of reactions
 * @param species_names List of species names
 * @return Tensor of reaction rates with shape (..., N_reactions)
 */
void calculate_reaction_rates(
    const torch::Tensor& T,
    const torch::Tensor& P,
    const torch::Tensor& C,
    torch::Tensor& rates,
    const std::vector<Reaction>& reactions,
    const std::vector<std::string>& species_names);

/**
 * @brief Calculate Jacobian matrix of reaction rates with respect to species concentrations
 * @param T Temperature tensor [K]
 * @param P Pressure tensor [Pa]
 * @param C Concentration tensor [kmol/m³] with shape (..., N_species)
 * @param reactions List of reactions
 * @param species_names List of species names
 * @return Tensor of Jacobians with shape (..., N_reactions, N_species)
 */
void calculate_jacobian(
    const torch::Tensor& T,
    const torch::Tensor& P,
    const torch::Tensor& C,
    torch::Tensor& jacobian,
    const std::vector<Reaction>& reactions,
    const std::vector<std::string>& species_names);

/**
 * @brief Class to handle chemical kinetics calculations
 * TODO: Convert the inputs to a matrix of orders. Not sure if this can be general.
 */
class Kinetics {
public:
    /**
     * @brief Construct a new Kinetics object
     * @param reactions List of chemical reactions
     * @param species_names List of species names
     */
    Kinetics(
        const std::vector<Reaction>& reactions,
        const std::vector<std::string>& species_names
    ) : reactions_(reactions), species_names_(species_names) {
    }

    /**
     * @brief Calculate reaction rates
     * @param T Temperature tensor [K]
     * @param P Pressure tensor [Pa] 
     * @param C Concentration tensor [kmol/m³] with shape (..., N_species)
     * @param rates Output tensor of reaction rates with shape (..., N_reactions)
     */
    void eval_rates(
        const torch::Tensor& T,
        const torch::Tensor& P,
        const torch::Tensor& C,
        torch::Tensor& rates
    ) {
        calculate_reaction_rates(T, P, C, rates, reactions_, species_names_);
    }

    /**
     * @brief Calculate Jacobian matrix of reaction rates
     * @param T Temperature tensor [K]
     * @param P Pressure tensor [Pa]
     * @param C Concentration tensor [kmol/m³] with shape (..., N_species)
     * @param jacobian Output tensor of Jacobians with shape (..., N_reactions, N_species)
     */
    void eval_jacobian(
        const torch::Tensor& T,
        const torch::Tensor& P,
        const torch::Tensor& C,
        torch::Tensor& jacobian
    ) {
        calculate_jacobian(T, P, C, jacobian, reactions_, species_names_);
    }

private:
    std::vector<Reaction> reactions_;
    std::vector<std::string> species_names_;
};
} // namespace kintera 