// kintera
#include <kintera/constants.h>

#include "eval_uh.hpp"
#include "thermo.hpp"
#include "thermo_dispatch.hpp"

namespace kintera {

ThermoXImpl::ThermoXImpl(const ThermoOptions &options_) : options(options_) {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  if (options.mu_ratio().empty()) {
    options.mu_ratio() = std::vector<double>(nvapor + ncloud, 1.);
  }

  if (options.cref_R().empty()) {
    options.cref_R() = std::vector<double>(nvapor + ncloud, 5. / 2.);
  }

  if (options.uref_R().empty()) {
    options.uref_R() = std::vector<double>(nvapor + ncloud, 0.);
  }

  // populate higher-order thermodynamic functions
  while (options.intEng_R_extra().size() < options.species().size()) {
    options.intEng_R_extra().push_back(nullptr);
  }

  while (options.cv_R_extra().size() < options.species().size()) {
    options.cv_R_extra().push_back(nullptr);
  }

  while (options.cp_R_extra().size() < options.species().size()) {
    options.cp_R_extra().push_back(nullptr);
  }

  while (options.czh().size() < options.species().size()) {
    options.czh().push_back(nullptr);
  }

  while (options.czh().size() < options.species().size()) {
    options.czh_ddC().push_back(nullptr);
  }

  reset();
}

void ThermoXImpl::reset() {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  TORCH_CHECK(options.mu_ratio().size() == nvapor + ncloud,
              "mu_ratio size mismatch");
  TORCH_CHECK(options.cref_R().size() == nvapor + ncloud,
              "cref_R size mismatch");
  TORCH_CHECK(options.uref_R().size() == nvapor + ncloud,
              "uref_R size mismatch");

  auto mud = constants::Rgas / options.Rd();
  mu =
      register_buffer("mu", torch::tensor(options.mu_ratio(), torch::kFloat64));

  auto cp_R = torch::tensor(options.cref_R(), torch::kFloat64);
  cp_R.narrow(0, 0, nvapor) += 1.;

  auto href_R = torch::tensor(options.uref_R(), torch::kFloat64);
  href_R.narrow(0, 0, nvapor) += options.Tref();

  // J/mol/K
  cp0 = register_buffer("cp0", cp_R * constants::Rgas);
  h0 = register_buffer("h0", href_R * constants::Rgas - cp0 * options.Tref());

  // populate stoichiometry matrix
  int nspecies = options.species().size();
  int nreact = options.react().size();

  stoich = register_buffer("stoich",
                           torch::zeros({nspecies, nreact}, torch::kFloat64));

  for (int j = 0; j < options.react().size(); ++j) {
    auto const &r = options.react()[j];
    for (int i = 0; i < options.species().size(); ++i) {
      auto it = r.reaction().reactants().find(options.species()[i]);
      if (it != r.reaction().reactants().end()) {
        stoich[i][j] = -it->second;
      }
      it = r.reaction().products().find(options.species()[i]);
      if (it != r.reaction().products().end()) {
        stoich[i][j] = it->second;
      }
    }
  }
}

torch::Tensor ThermoXImpl::compute(
    std::string ab, std::initializer_list<torch::Tensor> args) const {
  if (ab == "X->Y") {
    _X.resize_as_(*args.begin());
    _X.copy_(*args.begin());
    _xfrac_to_yfrac(_X, _Y);
  } else if (ab == "V->D") {
    _V.resize_as_(*args.begin());
    _V.copy_(*args.begin());
    _conc_to_dens(_V, _D);
  } else if (ab == "VT->cp") {
    _V.resize_as_(*args.begin());
    _V.copy_(*args.begin());
    _T.resize_as_(*(args.begin() + 1));
    _T.copy_(*(args.begin() + 1));
    _cp_mean(_V, _T, _cp);
  } else if (ab == "VT->H") {
    _V.resize_as_(*args.begin());
    _V.copy_(*args.begin());
    _T.resize_as_(*(args.begin() + 1));
    _T.copy_(*(args.begin() + 1));
    _temp_to_enthalpy(_V, _T, _H);
  } else if (ab == "TPX->V") {
    _T.resize_as_(*args.begin());
    _T.copy_(*args.begin());
    _P.resize_as_(*(args.begin() + 1));
    _P.copy_(*(args.begin() + 1));
    _X.resize_as_(*(args.begin() + 2));
    _X.copy_(*(args.begin() + 2));
    _xfrac_to_conc(_T, _P, _X, _V);
  } else if (ab == "TPX->D") {
    out =
        _temp_to_dens(*args.begin(), *(args.begin() + 1), *(args.begin() + 2));
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

torch::Tensor ThermoXImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                   torch::Tensor xfrac) {
  auto xfrac0 = xfrac.clone();

  // prepare data
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(false)
                  .declare_static_shape(xfrac.sizes(),
                                        /*squash_dims=*/{xfrac.dim() - 1})
                  .add_output(xfrac)
                  .add_owned_input(temp.unsqueeze(-1))
                  .add_owned_input(pres.unsqueeze(-1))
                  .add_owned_input(stoich)
                  .build();

  // prepare svp function
  user_func1 *logsvp_func = new user_func1[options.react().size()];
  for (int i = 0; i < options.react().size(); ++i) {
    logsvp_func[i] = options.react()[i].func();
  }

  // call the equilibrium solver
  at::native::call_equilibrate_tp(xfrac.device().type(), iter,
                                  options.vapor_ids().size() + 1, logsvp_func,
                                  options.ftol(), options.max_iter());

  delete[] logsvp_func;

  return xfrac - xfrac0;
}

torch::Tensor ThermoXImpl::_temp_to_dens(torch::Tensor temp, torch::Tensor pres,
                                         torch::Tensor xfrac) const {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();
  int nmass = nvapor + ncloud;

  auto xgas = 1. - xfrac.narrow(-1, 1 + nvapor, ncloud).sum(-1);
  auto ftv = xgas / (1. + xfrac.narrow(-1, 1, nmass).matmul(mu_ratio_m1));
  return pres / (temp * ftv * options.Rd());
}

torch::Tensor ThermoXImpl::_xfrac_to_yfrac(torch::Tensor xfrac) const {
  int nmass = xfrac.size(-1) - 1;

  auto vec = xfrac.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i + 1] = xfrac.size(i);
  }
  vec[0] = nmass;

  auto yfrac = torch::empty(vec, xfrac.options());

  // (..., nmass + 1) -> (nmass, ...)
  int ndim = xfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  yfrac.permute(vec) = xfrac.narrow(-1, 1, nmass) * (mu_ratio_m1 + 1.);
  auto sum = 1. + xfrac.narrow(-1, 1, nmass).matmul(mu_ratio_m1);
  return yfrac / sum.unsqueeze(0);
}

torch::Tensor ThermoXImpl::_xfrac_to_conc(torch::Tensor temp,
                                          torch::Tensor pres,
                                          torch::Tensor xfrac) const {
  auto nvapor = options.vapor_ids().size();
  auto ncloud = options.cloud_ids().size();

  auto gas_conc = pres / (temp * constants::Rgas);
  auto xgas = 1. - xfrac.narrow(-1, 1 + nvapor, ncloud).sum(-1);

  auto conc = torch::empty_like(xfrac);
  conc.narrow(-1, 0, 1 + nvapor) =
      gas_conc * xfrac.narrow(-1, 0, 1 + nvapor) / xgas.unsqueeze(-1);
  conc.narrow(-1, 1 + nvapor, ncloud) = xfrac.narrow(-1, 1 + nvapor, ncloud) /
                                        xfrac.select(-1, 0).unsqueeze(-1) *
                                        conc.select(-1, 0).unsqueeze(-1);

  return conc;
}

torch::Tensor ThermoXImpl::_temp_to_enthalpy(
    torch::Tensor temp, torch::Tensor conc,
    torch::optional<torch::Tensor> out) const {
  int ngas = 1 + options.vapor_ids().size();
  auto intEng = eval_intEng_R(temp, conc, options) * constants::Rgas;
  auto zRT = eval_czh(temp, conc, options).narrow(-1, 0, ngas) *
             constants::Rgas * temp.unsqueeze(-1);
  int ngas = 1 + options.vapor_ids().size();

  if (out.has_value()) {
    out = (intEng * conc).sum(-1) + zRT * conc.narrow(-1, 0, ngas);
    return out.value();
  } else {
    return (intEng * conc).sum(-1) + zRT * conc.narrow(-1, 0, ngas);
  }
}

torch::Tensor ThermoXImpl::_cp_vol(torch::Tensor temp, torch::Tensor conc,
                                   torch::optional<torch::Tensor> out) const {
  auto cp1 = eval_cp_R(temp, conc, options) * constants::Rgas;
  auto ctotal = conc.sum(-1);

  if (out.has_value()) {
    out = (cp1 * conc).sum(-1) / ctotal;
    return out.value();
  } else {
    return (cp1 * conc).sum(-1) / ctotal;
  }
}

}  // namespace kintera
