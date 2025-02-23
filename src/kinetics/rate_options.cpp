// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include "rate_options.hpp"

namespace kintera {

RateOptions RateOptions::from_yaml(const YAML::Node& node) {
  RateOptions options;

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

RateOptions RateOptions::from_map(
    const std::map<std::string, std::string>& param) {
  RateOptions options;

  if (param.count("A")) {
    options.A(std::stod(param.at("A")));
  }

  if (param.count("b")) {
    options.b(std::stod(param.at("b")));
  }

  if (param.count("Ea")) {
    options.Ea_R(std::stod(param.at("Ea")));
  }

  if (param.count("E4")) {
    options.E4_R(std::stod(param.at("E4")));
  }

  if (param.count("order")) {
    options.order(std::stod(param.at("order")));
  }

  return options;

}  // namespace kintera
