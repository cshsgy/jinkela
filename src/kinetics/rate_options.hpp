#pragma once

// kintera
#include <kintera/add_arg.h>

class YAML::Node;

namespace kintera {

//! Options to initialize all reaction rate constants
struct RateOptions {
  static RateOptions from_yaml(const YAML::Node& node);
  static RateOptions from_map(const std::map<std::string, std::string>& param);

  //! Pre-exponential factor. The unit system is (kmol, m, s);
  //! actual units depend on the reaction order
  ADD_ARG(double, A) = 1.;

  //! Dimensionless temperature exponent
  ADD_ARG(double, b) = 0.;

  //! Activation energy in K
  ADD_ARG(double, Ea_R) = 1.;

  //! Additional 4th parameter in the rate expression
  ADD_ARG(double, E4_R) = 0.;

  //! Reaction order (non-dimensional)
  ADD_ARG(double, order) = 1;
};

}  // namespace kintera
