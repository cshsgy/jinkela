// kintera
#include "kinetic_rate.hpp"

namespace kintera {

KineticRateImpl::KineticRateImpl(const KineticRateOptions& options_)
    : options(options_) {
  reset();
}

KineticRateImpl::reset() {
  std::pair<std::vector<int>> indices;
  std::vector<double> order_values;
  std::vector<double> stoich_values;

  for (const auto& rxn : options.reactions()) {
    int rxn_id = std::distance(options.reactions().begin(), rxn);

    for (const auto& [species, stoich] : rxn.reactants()) {
      int species_id = std::distance(options.species().begin(), species);

      indices.first.push_back(rxn_id);
      indices.second.push_back(species_id);
      stoich_values.push_back(stoich);

      if (rxn.orders().find(species) != rxn.orders().end()) {
        order_values.push_back(rxn.orders()[species]);
      } else {
        order_values.push_back(1.0);
      }
    }

    for (const auto& [species, stoich] : rxn.products()) {
      int species_id = std::distance(options.species().begin(), species);

      indices.first.push_back(rxn_id);
      indices.second.push_back(species_id);
      stoich_values.push_back(-stoich);
    }

    // reversible reaction?
  }

  order = register_buffer(
      "order", torch::sparse_coo_tensor(
                   indices, order_values,
                   {options.reactions().size(), options.species().size()}));

  stoich = register_buffer(
      "stoich", torch::sparse_coo_tensor(
                    indices, stoich_values,
                    {options.reactions().size(), options.species().size()}));
}

torch::Tensor KineticRateImpl::forward(torch::Tensor conc,
                                       torch::Tensor log_rate_constant) {
  int nreaction = order.size(0);
  int nspecies = order.size(1);
  return (order.view({1, 1, nreaction, nspecies}).matmul(conc.log()) +
          log_rate_constant)
      .exp();
}

}  // namespace kintera
