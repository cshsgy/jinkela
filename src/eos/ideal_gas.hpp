#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <add_arg.h>

namespace kintera {

struct EquationOfStateOptions {
  EquationOfStateOptions() = default;

  ADD_ARG(double, mu) = 2.0e-3;
  ADD_ARG(double, gamma_ref) = 1.4;
};

class IdealGasEOSImpl : public torch::nn::Cloneable<IdealGasEOSImpl> {
 public:
  //! options with which this `IdealGasEOSImpl` was constructed
  EquationOfStateOptions options;

  IdealGasEOSImpl() = default;
  explicit IdealGasEOSImpl(const EquationOfStateOptions& options_);

  //! \brief Calculate internal energy from density and pressure
  torch::Tensor get_intEng(torch::Tensor rho, torch::Tensor pres) {
    return pres / (options.gamma_ref() - 1.) / rho;
  }

  //! \brief Calculate pressure from density and internal energy
  torch::Tensor get_pres(torch::Tensor rho, torch::Tensor intEng) {
    return (options.gamma_ref() - 1.) * intEng * rho;
  }

  //! \brief Calculate temperature from density and pressure
  torch::Tensor get_temp(torch::Tensor rho, torch::Tensor pres) {
    return (pres * options.mu()) / (rho * constants::Rgas);
  }
};
TORCH_MODULE(IdealGasEOS);

}  // namespace kintera
