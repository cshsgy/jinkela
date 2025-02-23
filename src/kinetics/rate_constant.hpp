#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/reaction.hpp>

namespace kintera {

struct RateConstantOptions {
  ADD_ARG(std::vector<std::string>, types) = {};
  ADD_ARG(std::vector<RateEvaluatorOptions>, rate_opts) = {};
};

class RateConstantImpl : public torch::nn::Module {
 public:
  //! evaluate reaction rate coefficients
  std::vector<torch::nn::AnyModule> evals;

  //! options with which this `RateConstantImpl` was constructed
  RateConstantOptions options;

  //! Constructor to initialize the layer
  RateConstantImpl() = default;
  explicit RateConstantImpl(const RateConstantOptions& options_);
  void reset() override;

  //! Compute species rate of change
  /*!
   * \param T temperature [K], shape (ncol, nlyr)
   * \param other other parameters
   * \return log rate constant in log(kmol, m, s), shape (ncol, nlyr, nreaction)
   */
  torch::Tensor forward(torch::Tensor T,
                        std::map<std::string, torch::Tensor> const& other);
};

TORCH_MODULE_IMPL(RateConstant, RateConstantImpl);

}  // namespace kintera
