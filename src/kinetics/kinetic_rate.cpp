// kintera
#include "kinetic_rate.hpp"

namespace kintera {

KineticRateImpl::KineticRateImpl(const KineticRateOptions& options_)
    : options(options_) {
  reset();
}

void KineticRateImpl::reset() {
  int nreaction = options.reactions().size();
  int nspecies = options.species().size();

  order = register_buffer("order", torch::zeros({nreaction, nspecies}));
  stoich = register_buffer("stoich", torch::zeros({nreaction, nspecies}));

  // go through the reactants
  for (auto it = options.reactions().begin(); it != options.reactions().end();
       ++it) {
    int rxn_id = std::distance(options.reactions().begin(), it);

    for (const auto& [species, coeff] : it->reactants()) {
      auto jt = std::find(options.species().begin(), options.species().end(),
                          species);
      TORCH_CHECK(jt != options.species().end(), "Species ", species,
                  " not found in species list");

      int species_id = std::distance(options.species().begin(), jt);
      stoich[rxn_id][species_id] -= coeff;

      if (it->orders().find(species) != it->orders().end()) {
        order[rxn_id][species_id] = it->orders().at(species);
      } else {
        order[rxn_id][species_id] = 1.;
      }
    }

    for (const auto& [species, coeff] : it->products()) {
      auto jt = std::find(options.species().begin(), options.species().end(),
                          species);
      TORCH_CHECK(jt != options.species().end(), "Species ", species,
                  " not found in species list");
      int species_id = std::distance(options.species().begin(), jt);
      stoich[rxn_id][species_id] += coeff;
    }

    // reversible reaction?
  }
}

torch::Tensor KineticRateImpl::forward(torch::Tensor conc,
                                       torch::Tensor log_rate_constant) {
  int nreaction = order.size(0);
  int nspecies = order.size(1);
  return (order.view({1, 1, nreaction, -1})
              .matmul(conc.unsqueeze(-1).log())
              .squeeze(-1) +
          log_rate_constant)
      .exp();
}

}  // namespace kintera
