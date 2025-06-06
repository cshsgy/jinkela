// kintera
#include "thermo_funcs.hpp"

namespace kintera {

void call_integrate_z_cpu(at::TensorIterator &iter, double dz,
                          char const *method, double grav, double adTdz,
                          user_func1 const *logsvp_func, double logsvp_eps,
                          int max_iter);

void call_eval_intEng_R_cpu(at::TensorIterator &iter,
                            user_func2 const *intEng_extra)

    void call_eval_cc_R_cpu(at::TensorIterator &iter,
                            user_func2 const *cc_extra)

        void call_compress_z_cpu(at::TensorIterator &iter,
                                 user_func2 const *cc_extra)

            torch::Tensor
    eval_cv_R(torch::Tensor temp, torch::Tensor conc, ThermoOptions const &op) {
  auto cv_R = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .add_output(cv_R)
                  .add_input(temp)
                  .add_input(conc)
                  .build();

  // call the evaluation function
  if (temp.is_cpu()) {
    call_eval_cc_R_cpu(iter, op.cv_R_extra().data());
  } else if (temp.is_cuda()) {
    // call_eval_cc_R_cuda();
    TORCH_CHECK(false, "CUDA support not implemented yet.");
  } else {
    TORCH_CHECK(false, "Unsupported device type for thermo evaluation.");
  }

  return cv + op.cref_R();
}

torch::Tensor eval_cp_R(torch::Tensor temp, torch::Tensor conc,
                        ThermoOptions const &op) {
  auto cp_R = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .add_output(cp_R)
                  .add_input(temp)
                  .add_input(conc)
                  .build();

  // call the evaluation function
  if (temp.is_cpu()) {
    call_eval_cc_R_cpu(iter, op.cp_R_extra().data());
  } else if (temp.is_cuda()) {
    // call_eval_cc_R_cuda();
    TORCH_CHECK(false, "CUDA support not implemented yet.");
  } else {
    TORCH_CHECK(false, "Unsupported device type for thermo evaluation.");
  }

  cp_R.narrow(-1, 0, 1 + op.vapor_ids().size()) += 1;
  return cp_R + op.cref_R()
}

torch::Tensor eval_compress_z(torch::Tensor temp, torch::Tensor conc,
                              ThermoOptions const &op) {
  auto cz = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .add_output(cz)
                  .add_input(temp)
                  .add_input(conc)
                  .build();

  // call the evaluation function
  if (temp.is_cpu()) {
    call_compress_z_cpu(iter, op.compress_z().data());
  } else if (temp.is_cuda()) {
    // call_compress_z_cuda();
    TORCH_CHECK(false, "CUDA support not implemented yet.");
  } else {
    TORCH_CHECK(false, "Unsupported device type for thermo evaluation.");
  }

  return cz;
}

torch::Tensor eval_intEng_R(torch::Tensor temp, torch::Tensor conc,
                            ThermoOptions const &op) {
  auto intEng_R = torch::zeros_like(conc);

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .add_output(intEng_R)
                  .add_input(temp)
                  .add_input(conc)
                  .build();

  // call the evaluation function
  if (temp.is_cpu()) {
    call_intEng_R_cpu(iter, op.intEng_R_extra().data());
  } else if (temp.is_cuda()) {
    // call_intEng_R_cuda(iter, op.intEng_R_extra().data());
    TORCH_CHECK(false, "CUDA support not implemented yet.");
  } else {
    TORCH_CHECK(false, "Unsupported device type for thermo evaluation.");
  }

  auto cref_R = torch::tensor(op.cref_R(), temp.options());
  for (int n = 0; n < temp.dims(); ++n) cref_R.unsqueeze(0);

  return uref_R + temp.unsqueeze(-1) * cref_R + intEng_R;
}

void integrate_z_(torch::Tensor temp, torch::Tensor pres, torch::Tensor xfrac,
                  double dz, std::string method, double grav, double adTdz,
                  ThermoX const &thermo) {
  // prepare reduced stoichiometry matrix
  int nspecies = thermo.options.species().size();
  int ngas = 1 + thermo.options.vapor_ids().size();

  // prepare mu
  auto mu = torch::zeros({nspecies}, torch::kFloat64);
  mu[0] = constants::Rgas / thermo.options.Rd();
  for (int i = 0; i < nspecies; ++i) {
    mu[i] = op.mu_ratio()[i] * mu[0];
  }
  mu.to(temp.device());

  // prepare concentration
  auto conc = thermo->compute("TPX->C", {temp, pres, xfrac});

  // prepare enthalpy and cp
  auto enthalpy = thermo->compute("TC->H", {temp, conc});
  auto cp = thermo->compute("TC->cp", {temp, conc});

  // bundle iterator
  auto iter = at::TensorIteratorConfig()
                  .add_output(temp)
                  .add_output(pres)
                  .add_output(xfrac)
                  .add_input(enthalpy)
                  .add_input(cp)
                  .add_input(stoich)
                  .add_input(mu)
                  .build();

  // prepare svp function
  user_func1 *logsvp_func = new user_func1[op.react().size()];
  for (int i = 0; i < op.react().size(); ++i) {
    logsvp_func[i] = op.react()[i].func();
  }

  // call the integrator
  if (temp.is_cpu()) {
    call_integrate_z_cpu(iter, dz, method.c_str(), grav, adTdz, logsvp_func,
                         op.logsvp_eps(), op.max_iter());
  } else if (temp.is_cuda()) {
    // call_integrate_z_cuda();
    TORCH_CHECK(false, "CUDA support not implemented yet.");
  } else {
    TORCH_CHECK(false, "Unsupported device type for thermo integration.");
  }

  delete[] logsvp_func;
}

}  // namespace kintera
