#include "species_rate.hpp"

namespace kintera {

torch::Tensor species_rate(torch::Tensor kinetic_rate, torch::Tensor stoich) {
  int nreaction = stoich.size(0);
  int nspecies = stoich.size(1);
  return kinetic_rate.matmul(stoich.view({1, 1, nreaction, nspecies}));
}

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

} // namespace kintera 