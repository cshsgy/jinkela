// kintera
#include "kinetic_rate.hpp"

namespace kintera {

KineticRateImpl::KineticRateImpl(const KineticRateOptions& options_)
    : options(options_) {
  reset();
}

void KineticRateImpl::reset() {
  std::array<std::vector<int>, 2> indices;
  std::vector<double> order_values;
  std::vector<double> stoich_values;

  for (int rxn_id = 0; rxn_id < options.reactions().size(); rxn_id++) {
    auto const& rxn = options.reactions()[rxn_id];

    for (const auto& [species, stoich] : rxn.reactants()) {
      auto jt = std::find(options.species().begin(), options.species().end(),
                          species);
      int species_id = std::distance(options.species().begin(), jt);

      indices[0].push_back(rxn_id);
      indices[1].push_back(species_id);
      stoich_values.push_back(stoich);

      if (rxn.orders().find(species) != rxn.orders().end()) {
        order_values.push_back(rxn.orders().at(species));
      } else {
        order_values.push_back(1.0);
      }
    }

    for (const auto& [species, stoich] : rxn.products()) {
      auto jt = std::find(options.species().begin(), options.species().end(),
                          species);
      int species_id = std::distance(options.species().begin(), jt);

      indices[0].push_back(rxn_id);
      indices[1].push_back(species_id);
      stoich_values.push_back(-stoich);
    }

    // reversible reaction?
  }

  // flatten the indices
  std::vector<int> flat;
  for (auto const& idx : indices) {
    flat.insert(flat.end(), idx.begin(), idx.end());
  }

  order = register_buffer(
      "order",
      torch::sparse_coo_tensor(
          torch::tensor(flat).view({2, (int)indices[0].size()}),
          torch::tensor(order_values),
          {(int)options.reactions().size(), (int)options.species().size()}));

  stoich = register_buffer(
      "stoich",
      torch::sparse_coo_tensor(
          torch::tensor(flat).view({2, (int)indices[0].size()}),
          torch::tensor(stoich_values),
          {(int)options.reactions().size(), (int)options.species().size()}));
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
