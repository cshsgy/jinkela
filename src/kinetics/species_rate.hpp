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
torch::Tensor species_rate(torch::Tensor kinetic_rate, torch::Tensor stoich) {
  int nreaction = stoich.size(0);
  int nspecies = stoich.size(1);
  return kinetic_rate.matmul(stoich.view({1, 1, nreaction, nspecies}));
}

//! Compute Jacobian of species rates with respect to concentrations
/*!
 * \param conc concentrations of species [kmol/m³], shape (ncol, nlyr, nspecies)
 * \param reaction_rate reaction rates, shape (ncol, nlyr, nreactions)
 * \param stoich stoichiometry matrix, shape (nreaction, nspecies)
 * \param order order matrix, shape (nreaction, nspecies)
 * \return Jacobian matrix, shape (ncol, nlyr, nspecies, nspecies)
 */
torch::Tensor species_jacobian(torch::Tensor conc, torch::Tensor reaction_rate, 
                               torch::Tensor stoich, torch::Tensor order) {
  auto ncol = conc.size(0);
  auto nlyr = conc.size(1);
  auto nspecies = conc.size(2);
  auto nreaction = reaction_rate.size(2);
  
  auto jac = torch::zeros({ncol, nlyr, nspecies, nspecies}, conc.options());
  
  for (int64_t i = 0; i < nspecies; ++i) {
    for (int64_t j = 0; j < nspecies; ++j) {
      for (int64_t r = 0; r < nreaction; ++r) {
        if (order.index({r, j}).item<double>() != 0.0) {
          auto drdc = order.index({r, j}).item<double>() * reaction_rate.select(2, r) / conc.select(2, j);
          jac.index({torch::indexing::Slice(), torch::indexing::Slice(), i, j}) += 
              stoich.index({r, i}).item<double>() * drdc;
        }
      }
    }
  }
  
  return jac;
}

}  // namespace kintera
