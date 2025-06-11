// kintera
#include "eval_uh.hpp"

#include "thermo_dispatch.hpp"

namespace kintera {

torch::Tensor eval_cv_R(torch::Tensor temp, torch::Tensor conc,
                        ThermoOptions const& op) {
  auto cv_R_extra = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .declare_static_shape(cv_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cv_R_extra)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cv_R_extra.device().type(), iter,
                           op.cv_R_extra().data());

  auto cref_R = torch::tensor(op.cref_R(), temp.options());

  return cref_R + cv_R_extra;
}

torch::Tensor eval_cp_R(torch::Tensor temp, torch::Tensor conc,
                        ThermoOptions const& op) {
  auto cp_R_extra = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .declare_static_shape(cp_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cp_R_extra)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cp_R_extra.device().type(), iter,
                           op.cp_R_extra().data());

  auto cref_R = torch::tensor(op.cref_R(), temp.options());
  cref_R.narrow(-1, 0, 1 + op.vapor_ids().size()) += 1;

  return cref_R + cp_R_extra;
}

torch::Tensor eval_czh(torch::Tensor temp, torch::Tensor conc,
                       ThermoOptions const& op) {
  auto cz = torch::zeros_like(conc);
  cz.narrow(-1, 0, 1 + op.vapor_ids().size()) = 1.;

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .declare_static_shape(cz.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cz)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cz.device().type(), iter, op.czh().data());

  return cz;
}

torch::Tensor eval_czh_ddC(torch::Tensor temp, torch::Tensor conc,
                           ThermoOptions const& op) {
  auto cz_ddC = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .declare_static_shape(cz_ddC.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(cz_ddC)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cz_ddC.device().type(), iter, op.czh_ddC().data());

  return cz_ddC;
}

torch::Tensor eval_intEng_R(torch::Tensor temp, torch::Tensor conc,
                            ThermoOptions const& op) {
  auto intEng_R_extra = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .declare_static_shape(intEng_R_extra.sizes(),
                                        /*squash_dim=*/{conc.dim() - 1})
                  .add_output(intEng_R_extra)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(intEng_R.device().type(), iter,
                           op.intEng_R_extra().data());

  auto cref_R = torch::tensor(op.cref_R(), temp.options());
  auto uref_R = torch::tensor(op.uref_R(), temp.options());

  return uref_R + temp.unsqueeze(-1) * cref_R + intEng_R_extra;
}

torch::Tensor eval_enthalpy_R(torch::Tensor temp, torch::Tensor conc,
                              ThermoOptions const& op) {
  int ngas = 1 + op.vapor_ids().size();
  int ncloud = op.cloud_ids().size();

  auto enthalpy_R = torch::zeros_like(conc);
  auto czh = eval_czh(temp, conc, op);

  enthalpy_R.narrow(-1, 0, ngas) = eval_intEng_R(temp, conc, op) +
                                   czh.narrow(-1, 0, ngas) * temp.unsqueeze(-1);

  auto cref_R = torch::tensor(op.cref_R(), temp.options());
  auto uref_R = torch::tensor(op.uref_R(), temp.options());
  enthalpy_R.narrow(-1, ngas, ncloud) =
      uref_R.narrow(-1, ngas, ncloud) +
      cref_R.narrow(-1, ngas, ncloud) * temp.unsqueeze(-1) +
      czh.narrow(-1, ngas, ncloud);

  return enthalpy_R;
}

}  // namespace kintera
