// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/utils/constants.hpp>

#include "arrhenius.hpp"

namespace kintera {

ArrheniusOptions ArrheniusOptions::from_yaml(const YAML::Node& root) {
  ArrheniusOptions options;

  for (auto const& rxn_node : root) {
    if (!rxn_node["type"]) {
      TORCH_CHECK(false, "Reaction type not specified");
    }

    if (rxn_node["type"].as<std::string>() != "Arrhenius") {
      continue;
    }

    auto node = rxn_node["rate-constant"];
    if (node["A"]) {
      options.A().push_back(node["A"].as<double>());
    } else {
      options.A().push_back(1.);
    }

    if (node["b"]) {
      options.b().push_back(node["b"].as<double>());
    } else {
      options.b().push_back(0.);
    }

    if (node["Ea"]) {
      options.Ea_R().push_back(node["Ea"].as<double>());
    } else {
      options.Ea_R().push_back(1.);
    }

    if (node["E4"]) {
      options.E4_R().push_back(node["E4"].as<double>());
    } else {
      options.E4_R().push_back(0.);
    }
  }

  return options;
}

ArrheniusOptions ArrheniusOptions::from_map(
    const std::vector<std::map<std::string, std::string>>& params) {
  ArrheniusOptions options;

  for (auto const& param : params) {
    if (param.count("A")) {
      options.A().push_back(std::stod(param.at("A")));
    } else {
      options.A().push_back(1.);
    }

    if (param.count("b")) {
      options.b().push_back(std::stod(param.at("b")));
    } else {
      options.b().push_back(0.);
    }

    if (param.count("Ea")) {
      options.Ea_R().push_back(std::stod(param.at("Ea")));
    } else {
      options.Ea_R().push_back(1.);
    }

    if (param.count("E4")) {
      options.E4_R().push_back(std::stod(param.at("E4")));
    } else {
      options.E4_R().push_back(0.);
    }
  }

  return options;
}

ArrheniusImpl::ArrheniusImpl(ArrheniusOptions const& options_)
    : options(options_) {
  reset();
}

void ArrheniusImpl::reset() {
  logA = register_buffer("logA", torch::tensor(options.A()).log());
  b = register_buffer("b", torch::tensor(options.b()));
  Ea_R = register_buffer("Ea_R", torch::tensor(options.Ea_R()));
  E4_R = register_buffer("E4_R", torch::tensor(options.E4_R()));
}

void ArrheniusImpl::pretty_print(std::ostream& os) const {
  os << "Arrhenius Rate: " << std::endl;

  for (size_t i = 0; i < options.A().size(); i++) {
    os << "(" << i + 1 << ") A = " << options.A()[i]
       << ", b = " << options.b()[i]
       << ", Ea = " << options.Ea_R()[i] * constants::GasConstant << " J/kmol"
       << std::endl;
  }
}

torch::Tensor ArrheniusImpl::forward(
    torch::Tensor T, std::map<std::string, torch::Tensor> const& other) {
  return logA.view({1, 1, -1}) + b.view({1, 1, -1}) * T.unsqueeze(-1).log() -
         Ea_R.view({1, 1, -1}) / T.unsqueeze(-1);
}

torch::Tensor ArrheniusImpl::jacobian(
    torch::Tensor conc, torch::Tensor reaction_rate, torch::Tensor stoich, 
    std::vector<std::string> const& species) {
  auto ncol = conc.size(0);
  auto nlyr = conc.size(1);
  auto nspecies = conc.size(2);
  
  auto jac = torch::zeros({ncol, nlyr, nspecies, nspecies}, conc.options());
  
  for (int64_t i = 0; i < nspecies; ++i) {
    for (int64_t j = 0; j < nspecies; ++j) {
      auto drdc = torch::zeros({ncol, nlyr, reaction_rate.size(2)}, conc.options());
      for (int64_t r = 0; r < reaction_rate.size(2); ++r) {
        auto reactant_stoich = stoich.index({0, r}).to(conc.device());
        auto product_stoich = stoich.index({1, r}).to(conc.device());
        auto net_stoich = product_stoich - reactant_stoich;
        
        if (reactant_stoich.index({j}).item<double>() != 0.0) {
          auto factor = reactant_stoich.index({j}).item<double>() / conc.select(2, j);
          drdc.select(2, r) = reaction_rate.select(2, r) * factor;
        }
      }
      
      for (int64_t r = 0; r < reaction_rate.size(2); ++r) {
        auto reactant_stoich = stoich.index({0, r}).to(conc.device());
        auto product_stoich = stoich.index({1, r}).to(conc.device());
        auto net_stoich = product_stoich - reactant_stoich;
        
        jac.index({torch::indexing::Slice(), torch::indexing::Slice(), i, j}) += 
            drdc.select(2, r) * net_stoich[i];
      }
    }
  }
  
  return jac;
}

/*torch::Tensor ArrheniusRate::ddTRate(torch::Tensor T, torch::Tensor P) const {
    return (m_Ea_R * 1.0 / T + m_b) * 1.0 / T;
}*/

}  // namespace kintera
