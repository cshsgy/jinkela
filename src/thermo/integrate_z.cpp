// kintera
#include "thermo_funcs.hpp"

namespace kintera {

void call_integrate_z_cpu(at::TensorIterator &iter, double dz,
                          char const *method, double grav, double adTdz,
                          user_func1 const *logsvp_func, double logsvp_eps,
                          int max_iter);

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
