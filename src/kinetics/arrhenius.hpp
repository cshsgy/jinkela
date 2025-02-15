#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/add_arg.h>

namespace kintera {

//! Options to initialize `Arrhenius`
struct ArrheniusOptions {
  static ArrheniusOptions from_yaml(const YAML::Node& node);

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

class ArrheniusImpl : public torch::nn::Cloneable<ArrheniusImpl> {
 public:
  //! options with which this `ArrheniusImpl` was constructed
  ArrheniusOptions options;

  //! Constructor to initialize the layer
  ArrheniusImpl() = default;
  explicit ArrheniusImpl(ArrheniusOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute the reaction rate constant
  /*!
   * \param T temperature [K]
   * \param P pressure [Pa]
   * \return reaction rate constant in (kmol, m, s)
   */
  torch::Tensor forward(torch::Tensor T,
                        torch::optional<torch::Tensor> P = torch::nullopt);
};
TORCH_MODULE(Arrhenius);

}  // namespace kintera
