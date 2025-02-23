// kintera
#include "species_rate.hpp"

namespace kintera {

SpeciesRateImpl::SpeciesRateImpl(const KineticsOptions& options_)
    : options(options_) {
  reset();
}

SpeciesRateImpl::reset() {
  std::pair<std::vector<int>> indices;
  std::vector<double> values;

  for (const auto& rxn : options.reactions()) {
    int rxn_id = std::distance(options.reactions().begin(), rxn);

    for (const auto& [species, stoich] : rxn.reactants()) {
      int species_id = std::distance(options.species().begin(), species);

      indices.first.push_back(rxn_id);
      indices.second.push_back(species_id);
      values.push_back(stoich);
    }

    for (const auto& [species, stoich] : rxn.products()) {
      int species_id = std::distance(options.species().begin(), species);

      indices.first.push_back(rxn_id);
      indices.second.push_back(species_id);
      values.push_back(-stoich);
    }

    // reversible reaction?
  }
}

torch::Tensor SpeciesRateImpl::forward(torch::Tensor conc,
                                       torch::Tensor kinetic_rate) {

}  // namespace kintera
