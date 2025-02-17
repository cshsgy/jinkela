#include "kinetics_rates.hpp"

// C++
#include <algorithm>
#include <stdexcept>

namespace kintera {

KineticsRatesImpl::KineticsRatesImpl(
    const std::map<Reaction, torch::nn::AnyModule>& reaction_rates,
    const torch::Tensor& stoich_matrix,
    const std::vector<std::string>& species)
    : reaction_rates_(reaction_rates), 
      stoich_matrix_(stoich_matrix),
      species_(species) {}

torch::Tensor KineticsRatesImpl::forward(torch::Tensor T, torch::Tensor P,
                                        torch::Tensor C) const {
  const auto n_reactions = stoich_matrix_.size(0);
  const auto n_species = stoich_matrix_.size(1);

  auto rate_shapes = C.sizes().vec();
  rate_shapes[0] = n_reactions;
  torch::Tensor rates = torch::zeros(rate_shapes, C.options());

  int64_t reaction_idx = 0;
  for (auto [reaction, rate_module] : reaction_rates_) {
    torch::Tensor rate = rate_module.forward(T, P);
    for (const auto& [species_name, order] : reaction.orders()) {
      auto it = std::find(species_.begin(), species_.end(), species_name);
      if (it == species_.end()) {
        throw std::runtime_error("Species " + species_name +
                               " not found in species list");
      }
      size_t j = std::distance(species_.begin(), it);
      rate = rate * torch::pow(C.select(0, j), order);
    }

    rates.index_put_({reaction_idx}, rate);
    reaction_idx++;
  }
  rates = rates.movedim(0, -1);
  auto result = torch::matmul(rates, stoich_matrix_);
  return result.movedim(-1, 0);
}

} // namespace kintera
