// yaml-cpp
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/reaction.hpp>

#include "arrhenius.hpp"
#include "rate_constant.hpp"

namespace kintera {

RateConstantImpl::RateConstantImpl(const RateConstantOptions& options_)
    : options(options_) {
  reset();
}

RateConstantImpl::reset() {
  YAML::Node root = YAML::LoadFile(filename);
  printf("Loading complete\n");
  for (auto const& rxn_node : root) {
    // if (rxn_node["efficiencies"]) {
    //     const auto& effs = rxn_node["efficiencies"];
    //     for (const auto& eff : effs) {
    //         std::string species = eff.first.as<std::string>();
    //         double value = eff.second.as<double>();
    //     }
    // }

    std::string type = "arrhenius";  // default type
    if (rxn_node["type"]) {
      type = rxn_node["type"].as<std::string>();
    }

    // TODO: Implement the support of other reaction types
    if (type == "arrhenius") {
      auto op = ArrheniusOptions::from_yaml(rxn_node["rate-constant"]);
      evals.push_back(torch::nn::AnyModule(Arrhenius(op)));
    } else if (type == "three-body") {
      TORCH_CHECK(false, "Three-body reaction not implemented");
    } else if (type == "falloff") {
      TORCH_CHECK(false, "Falloff reaction not implemented");
    } else {
      TORCH_CHECK(false, "Unknown reaction type: ", type);
    }
  }
}

torch::Tensor RateConstant::forward(
    torch::Tensor T, std::map<std::string, torch::Tensor> const& other) {
  auto shape = T.sizes().vec();
  shape.push_back(evals.size());

  torch::Tensor result = torch::empty(shape, T.options());

  for (int i = 0; i < evals.size(); i++) {
    result.select(-1, i) = evals[i].forward(T, other);
  }

  return result;
}

torch::Tensor KineticsRatesImpl::forward(torch::Tensor T, torch::Tensor P,
                                         torch::Tensor C) const {
  const auto n_reactions = stoich_matrix_.size(0);
  const auto n_species = stoich_matrix_.size(1);

  auto rate_shapes = C.sizes().vec();
  rate_shapes[0] = n_reactions;
  torch::Tensor rates = torch::zeros(rate_shapes, C.options());

  rates = rates.movedim(0, -1);
  auto result = torch::matmul(rates, stoich_matrix_);
  return result.movedim(-1, 0);
}

}  // namespace kintera
