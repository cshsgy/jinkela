#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/add_arg.h>

#include "rate_options.hpp"

namespace kintera {

class ArrheniusImpl : public torch::nn::Cloneable<ArrheniusImpl> {
 public:
  //! options with which this `ArrheniusImpl` was constructed
  ArrheniusOptions options;

  //! Constructor to initialize the layer
  ArrheniusImpl() = default;
  explicit ArrheniusImpl(RateOptions const& options_);
  void reset() override {}
  void pretty_print(std::ostream& os) const override;

  //! Compute the reaction rate constant
  /*!
   * \param T temperature [K]
   * \param other additional parameters
   * \return log reaction rate constant in ln(kmol, m, s)
   */
  torch::Tensor forward(torch::Tensor T,
                        std::map<std::string, torch::Tensor> const& other);
};
TORCH_MODULE(Arrhenius);

}  // namespace kintera
