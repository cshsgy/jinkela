// kintera
#include "eval_uh.hpp"

namespace kintera {

void call_func2_TC_cpu(at::TensorIterator& iter, user_func2 const* func);

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
  if (temp.is_cpu()) {
    call_func2_TC_cpu(iter, op.cv_R_extra().data());
  } else if (temp.is_cuda()) {
    // call_func2_TC_cuda();
    TORCH_CHECK(false, "CUDA support not implemented yet.");
  } else {
    TORCH_CHECK(false, "Unsupported device type for thermo evaluation.");
  }

  return cv + op.cref_R();
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
  if (temp.is_cpu()) {
    call_func2_TC_cpu(iter, op.cp_R_extra().data());
  } else if (temp.is_cuda()) {
    // call_func2_TC_cuda();
    TORCH_CHECK(false, "CUDA support not implemented yet.");
  } else {
    TORCH_CHECK(false, "Unsupported device type for thermo evaluation.");
  }

  cp_R.narrow(-1, 0, 1 + op.vapor_ids().size()) += 1;
  return cp_R + op.cref_R()
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
  if (temp.is_cpu()) {
    call_func2_TC_cpu(iter, op.compress_z().data());
  } else if (temp.is_cuda()) {
    // call_func2_TC_cuda();
    TORCH_CHECK(false, "CUDA support not implemented yet.");
  } else {
    TORCH_CHECK(false, "Unsupported device type for thermo evaluation.");
  }

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
  if (temp.is_cpu()) {
    call_func2_TC_cpu(iter, op.intEng_R_extra().data());
  } else if (temp.is_cuda()) {
    // call_func2_TC_cuda(iter, op.intEng_R_extra().data());
    TORCH_CHECK(false, "CUDA support not implemented yet.");
  } else {
    TORCH_CHECK(false, "Unsupported device type for thermo evaluation.");
  }

  auto cref_R = torch::tensor(op.cref_R(), temp.options());
  for (int n = 0; n < temp.dims(); ++n) cref_R.unsqueeze(0);

  return uref_R + temp.unsqueeze(-1) * cref_R + intEng_R;
}

}  // namespace kintera
