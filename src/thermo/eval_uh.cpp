// kintera
#include "eval_uh.hpp"

#include "thermo_dispatch.hpp"

namespace kintera {

torch::Tensor eval_cv_R(torch::Tensor temp, torch::Tensor conc,
                        ThermoOptions const& op) {
  auto cv_R = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .add_output(cv_R)
                  .add_input(temp)
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cv_R.device().type(), iter, op.cv_R_extra().data());

  auto vec = temp.sizes().vec();
  vec.push_back(op.cref_R().size());
  auto cref_R = torch::tensor(op.cref_R(), temp.options()).view(vec);

  return cv_R + cref_R.unsqueeze(-1) + 1;
}

torch::Tensor eval_cp_R(torch::Tensor temp, torch::Tensor conc,
                        ThermoOptions const& op) {
  auto cp_R = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .add_output(cp_R)
                  .add_input(temp)
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cp_R.device().type(), iter, op.cp_R_extra().data());

  auto vec = temp.sizes().vec();
  vec.push_back(op.cref_R().size());
  auto cref_R = torch::tensor(op.cref_R(), temp.options()).view(vec);

  cp_R.narrow(-1, 0, 1 + op.vapor_ids().size()) += 1;
  return cp_R + cref_R;
}

torch::Tensor eval_compress_z(torch::Tensor temp, torch::Tensor conc,
                              ThermoOptions const& op) {
  auto cz = torch::zeros_like(conc.narrow(-1, 0, 1 + op.vapor_ids().size()));

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .add_output(cz)
                  .add_input(temp)
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cz.device().type(), iter, op.compress_z().data());

  return cz;
}

torch::Tensor eval_intEng_R(torch::Tensor temp, torch::Tensor conc,
                            ThermoOptions const& op) {
  auto intEng_R = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .add_output(intEng_R)
                  .add_input(temp)
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(intEng_R.device().type(), iter,
                           op.intEng_R_extra().data());

  auto vec = temp.sizes().vec();
  vec.push_back(op.cref_R().size());

  auto cref_R = torch::tensor(op.cref_R(), temp.options()).view(vec);
  auto uref_R = torch::tensor(op.uref_R(), temp.options()).view(vec);

  return uref_R + temp.unsqueeze(-1) * cref_R + intEng_R;
}

}  // namespace kintera
