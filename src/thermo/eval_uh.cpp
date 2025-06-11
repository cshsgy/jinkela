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

  std::vector<int64_t> vec(temp.dim() + 1, 1);
  vec[temp.dim()] = op.cref_R().size();
  auto cref_R = torch::tensor(op.cref_R(), temp.options()).view(vec);

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

  std::vector<int64_t> vec(temp.dim() + 1, 1);
  vec[temp.dim()] = op.cref_R().size();
  auto cref_R = torch::tensor(op.cref_R(), temp.options()).view(vec);
  cref_R.narrow(-1, 0, 1 + op.vapor_ids().size()) += 1;
  return cref_R + cp_R_extra;
}

torch::Tensor eval_compress_z(torch::Tensor temp, torch::Tensor conc,
                              ThermoOptions const& op) {
  auto cz = torch::ones_like(conc.narrow(-1, 0, 1 + op.vapor_ids().size()));

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .declare_static_shape(cz.sizes(),
                                        /*squash_dim=*/{cs.dim() - 1})
                  .add_output(cz)
                  .add_owned_input(temp.unsqueeze(-1).expand_as(conc))
                  .add_input(conc)
                  .build();

  // call the evaluation function
  at::native::call_with_TC(cz.device().type(), iter, op.compress_z().data());

  return cz;
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

  std::vector<int64_t> vec(temp.dim() + 1, 1);
  vec[temp.dim()] = op.cref_R().size();

  auto cref_R = torch::tensor(op.cref_R(), temp.options()).view(vec);
  auto uref_R = torch::tensor(op.uref_R(), temp.options()).view(vec);

  return uref_R + temp.unsqueeze(-1) * cref_R + intEng_R_extra;
}

}  // namespace kintera
