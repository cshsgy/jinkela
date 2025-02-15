// kintera
#include "arrhenius.hpp"

#include <kintera/utils/constants.hpp>

namespace kintera {

ArrheniusOptions ArrheniusOptions::from_yaml(const YAML::Node& node) {
  ArrheniusOptions options;
  if (node["A"]) {
    options.A(node["A"].as<double>());
  }

  if (node["b"]) {
    options.b(node["b"].as<double>());
  }

  if (node["Ea"]) {
    options.Ea_R(node["Ea"].as<double>());
  }

  if (node["E4"]) {
    options.E4_R(node["E4"].as<double>());
  }

  if (node["order"]) {
    options.order(node["order"].as<double>());
  }

  return options;
}

ArrheniusImpl::ArrheniusImpl(ArrheniusOptions const& options_)
    : options(options_) {
  reset();
}

void ArrheniusImpl::pretty_print(std::ostream& os) const {
  os << "Arrhenius Rate: A = " << options.A() << ", b = " << options.b()
     << ", Ea = " << options.Ea_R() * constants::GasConstant << " J/kmol";
}

torch::Tensor ArrheniusImpl::forward(torch::Tensor T, torch::Tensor P) {
  return options.A() * (options.b() * T.log() - options.Ea_R() * 1.0 / T).exp();
}

/*torch::Tensor ArrheniusRate::ddTRate(torch::Tensor T, torch::Tensor P) const {
    return (m_Ea_R * 1.0 / T + m_b) * 1.0 / T;
}*/

}  // namespace kintera
