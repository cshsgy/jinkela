#pragma once

// torch
#include <torch/torch.h>

namespace kintera {

//! Compute species rate of change
/*!
 * \param stoich stoichiometry matrix, shape (nreaction, nspecies)
 * \param kinetic_rate kinetics rate of reactions [kmol/m^3/s],
 *        shape (ncol, nlyr, nreaction)
 */
torch::Tensor species_rate(torch::Tensor kinetic_rate, torch::Tensor stoich);

//! Compute Jacobian of species rates with respect to concentrations
/*!
 * \param conc concentrations of species [kmol/m³], shape (ncol, nlyr, nspecies)
 * \param reaction_rate reaction rates, shape (ncol, nlyr, nreactions)
 * \param stoich stoichiometry matrix, shape (nreaction, nspecies)
 * \param order order matrix, shape (nreaction, nspecies)
 * \return Jacobian matrix, shape (ncol, nlyr, nspecies, nspecies)
 */
torch::Tensor species_jacobian(torch::Tensor conc, torch::Tensor reaction_rate, 
                               torch::Tensor stoich, torch::Tensor order);

}  // namespace kintera
