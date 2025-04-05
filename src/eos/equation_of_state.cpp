#include "equation_of_state.hpp"

namespace kintera {

EquationOfStateImpl::EquationOfStateImpl(const EquationOfStateOptions& options_)
    : options(options_) {}

torch::Tensor EquationOfStateImpl::get_gamma(torch::Tensor rho,
                                             torch::Tensor intEng) const {
  torch::autograd::AutoGradMode guard(true);
  auto temp = get_temp(rho, get_pres(rho, intEng));

  auto grad_output = torch::ones_like(temp);
  auto dTdU =
      torch::autograd::grad({temp}, {intEng}, {grad_output}, true, true)[0];
  auto cv = 1. / dTdU;
  return 1. + constants::Rgas / (cv * options.mu());
}

}  // namespace kintera
