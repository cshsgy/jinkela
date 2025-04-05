// base
#include <configure.h>  // Index

// pyharp
#include <pyharp/constants.h>

// kintera
#include "thermo_formatter.hpp"
#include "thermodynamics.hpp"

namespace kintera {

ThermodynamicsOptions::ThermodynamicsOptions(ParameterInput pin) {
  gammad(pin->GetOrAddReal("thermodynamics", "gammad", 1.4));
  nvapor(pin->GetOrAddInteger("thermodynamics", "nvapor", 0));
  ncloud(pin->GetOrAddInteger("thermodynamics", "ncloud", 0));
}

ThermodynamicsImpl::ThermodynamicsImpl(const ThermodynamicsOptions& options_)
    : options(options_) {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  if (options.mu_ratio_m1().empty()) {
    options.mu_ratio_m1() = std::vector<double>(nvapor + ncloud, 0.);
  }

  if (options.cv_ratio_m1().empty()) {
    options.cv_ratio_m1() = std::vector<double>(nvapor + ncloud, 0.);
  }

  if (options.cp_ratio_m1().empty()) {
    options.cp_ratio_m1() = std::vector<double>(nvapor + ncloud, 0.);
  }

  if (options.h0().empty()) {
    options.h0() = std::vector<double>(1 + nvapor + ncloud, 0.);
  }

  reset();
}

void ThermodynamicsImpl::reset() {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  TORCH_CHECK(
      options.mu_ratio_m1().size() == options.nvapor() + options.ncloud(),
      "mu_ratio_m1 size mismatch");
  TORCH_CHECK(
      options.cv_ratio_m1().size() == options.nvapor() + options.ncloud(),
      "cv_ratio_m1 size mismatch");
  TORCH_CHECK(
      options.cp_ratio_m1().size() == options.nvapor() + options.ncloud(),
      "cp_ratio_m1 size mismatch");
  TORCH_CHECK(options.h0().size() == 1 + options.nvapor() + options.ncloud(),
              "h0 size mismatch");

  mu_ratio_m1 = register_buffer(
      "mu_ratio_m1", torch::tensor(options.mu_ratio_m1(), torch::kFloat64));

  cv_ratio_m1 = register_buffer(
      "cv_ratio_m1", torch::tensor(options.cv_ratio_m1(), torch::kFloat64));

  cp_ratio_m1 = register_buffer(
      "cp_ratio_m1", torch::tensor(options.cp_ratio_m1(), torch::kFloat64));

  h0 = register_buffer("h0", torch::tensor(options.h0(), torch::kFloat64));

  options.cond().species(options.species());
  pcond = register_module("cond", Condensation(options.cond()));
  options.cond() = pcond->options;
}

torch::Tensor ThermodynamicsImpl::f_eps(torch::Tensor yfrac) const {
  auto nvapor = options.vapor_ids().size();
  auto ncloud = options.cloud_ids().size();

  auto wu = yfrac.narrow(0, 0, nvapor).unfold(0, nvapor, 1);
  return 1. + wu.matmul(mu_ratio_m1.narrow(0, 0, nvapor)).squeeze(0) -
         yfrac.narrow(0, nvapor, ncloud).sum(0);
}

torch::Tensor ThermodynamicsImpl::f_sig(torch::Tensor yfrac) const {
  auto nmass = options.vapor_ids().size() + options.cloud_ids().size();

  auto wu = yfrac.narrow(0, 0, nmass).unfold(0, nmass, 1);
  return 1. + wu.matmul(cv_ratio_m1).squeeze(0);
}

torch::Tensor ThermodynamicsImpl::f_psi(torch::Tensor yfrac) const {
  auto nmass = options.vapor_ids().size() + options.cloud_ids().size();

  auto wu = yfrac.narrow(0, 0, nmass).unfold(0, nmass, 1);
  return 1. + wu.matmul(cp_ratio_m1).squeeze(0);
}

torch::Tensor ThermodynamicsImpl::get_mu() const {
  auto nmass = options.nvapor() + options.ncloud();

  auto result = torch::ones({1 + nmass}, mu_ratio_m1.options());
  result[0] = constants::Rgas / options.Rd();
  result.narrow(0, 1, nmass) = result[0] / (mu_ratio_m1 + 1.);

  return result;
}

torch::Tensor ThermodynamicsImpl::get_cv() const {
  auto nmass = options.nvapor() + options.ncloud();

  auto result = torch::empty({1 + nmass}, cv_ratio_m1.options());
  result[0] = options.Rd() / (options.gammad() - 1.);
  result.narrow(0, 1, namss) = result[0] * (cv_ratio_m1 + 1.);

  return result;
}

torch::Tensor ThermodynamicsImpl::get_cp() const {
  auto nmass = options.nvapor() + options.ncloud();

  auto result = torch::empty({1 + nmass}, cp_ratio_m1.options());
  result[0] = options.gammad() / (options.gammad() - 1.) * options.Rd();
  result.narrow(0, 1, nmass) = result[0] * (cp_ratio_m1 + 1.);

  return result;
}

torch::Tensor ThermodynamicsImpl::get_mole_fraction(torch::Tensor yfrac) const {
  int nmass = yfrac.size(0);
  TORCH_CHECK(nmass == options.vapor_ids().size() + options.cloud_ids().size(),
              "mass fraction size mismatch");

  auto vec = yfrac.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i] = yfrac.size(i + 1);
  }
  vec.back() = nmass + 1;

  auto xfrac = torch::empty(vec, yfrac.options());

  // (nmass, ...) -> (..., nmass)
  int ndim = yfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  xfrac.narrow(-1, 1, nmass) = yfrac.permute(vec) * (mu_ratio_m1 + 1.);
  auto sum = 1. + yfrac.permute(vec).matmul(mu_ratio_m1);
  xfrac.narrow(-1, 1, nmass) /= sum.unsqueeze(-1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, nmass).sum(-1);
  return xfrac;
}

torch::Tensor ThermodynamicsImpl::get_mass_fraction(torch::Tensor xfrac) const {
  int nmass = xfrac.size(-1) - 1;
  TORCH_CHECK(nmass == options.vapor_ids().size() + options.cloud_ids().size(),
              "mole fraction size mismatch");

  auto vec = xfrac.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i + 1] = xfrac.size(i);
  }
  vec[0] = nmass;

  auto yfrac = torch::empty(vec, xfrac.options());

  // (..., nmass) -> (nmass, ...)
  int ndim = xfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  yfrac.permute(vec) = xfrac.narrow(-1, 1, nmass) / (mu_ratio_m1 + 1.);
  auto sum =
      1. - xfrac.narrow(-1, 1, nmass).matmul(mu_ratio_m1 / (mu_ratio_m1 + 1.));
  return yfrac / sum.unsqueeze(0);
}

torch::Tensor ThermodynamicsImpl::forward(torch::Tensor rho,
                                          torch::Tensor yfrac,
                                          torch::Tensor intEng) {
  int nvapor = options.nvapor();
  int ncloud = options.ncloud();

  // total density
  auto rho = u[Index::IDN].clone();
  rho += u.narrow(0, Index::ICY, nvapor + ncloud).sum(0);

  auto feps = f_eps(yfrac);
  auto fsig = f_sig(yfrac);

  auto pres = (options.gammad_ref() - 1.) * u[Index::IEN] * feps / fsig;
  auto temp = pres / (rho * options.Rd() * feps);

  auto pres = peos->get_pres(rho, intEng);
  auto temp = peos->get_temp(rho, pres);
  auto conc = get_mole_concentration(yfrac);

  // std::cout << "pres = " << pres << std::endl;
  // std::cout << "temp = " << temp << std::endl;

  auto mu = get_mu();
  auto krate = torch::ones_like(temp);

  int iter = 0;
  for (; iter < options.max_iter(); ++iter) {
    /*std::cout << "iter = " << iter << std::endl;
    std::cout << "conc = " << conc << std::endl;*/
    auto intEng_RT = get_internal_energy_RT(temp);
    /*std::cout << "total intEng = "
              << (conc * intEng_RT).sum(-1) * constants::Rgas * temp
              << std::endl;*/

    auto cv_R = get_cv() * mu / constants::Rgas;

    auto rates = pcond->forward(temp, pres, conc, intEng_RT, cv_R, krate);
    // std::cout << "rates = " << rates << std::endl;
    // std::cout << "krate = " << krate << std::endl;

    conc += rates;
    TORCH_CHECK(conc.min().item<double>() >= 0., "negative concentration");

    auto dT = -temp * (rates * intEng_RT).sum(-1) / conc.matmul(cv_R);
    krate = where(dT * krate > 0., 2 * krate, torch::sign(dT));
    krate.clamp_(-options.boost(), options.boost());

    if (dT.abs().max().item<double>() < options.ftol()) break;
    temp += dT;
    pres = conc.narrow(-1, 0, 1 + nvapor).sum(-1) * constants::Rgas * temp;

    // std::cout << "temp = " << temp << std::endl;
  }

  auto u0 = u.clone();

  u.narrow(0, index::ICY, nvapor + ncloud) =
      (conc * mu).narrow(-1, 1, nvapor + ncloud).permute({3, 0, 1, 2});

  // total energy
  feps = f_eps(u / rho);
  fsig = f_sig(u / rho);
  u[index::IPR] = pres * fsig / feps / (options.gammad_ref() - 1.) + ke;

  return u - u0;
}

torch::Tensor ThermodynamicsImpl::equilibrate_tp(torch::Tensor temp,
                                                 torch::Tensor pres,
                                                 torch::Tensor yfrac) const {
  auto xfrac = get_mole_fraction(yfrac);

  int iter = 0;
  for (; iter < options.max_iter(); ++iter) {
    auto rates = pcond->equilibrate_tp(temp, pres, xfrac, 1 + options.nvapor());
    xfrac += rates;

    if ((rates / (xfrac + 1.e-10)).max().item<double>() < options.rtol()) break;
    TORCH_CHECK(xfrac.min().item<double>() >= 0., "negative mole fraction");
  }

  return get_mass_fraction(xfrac) - yfrac;
}

torch::Tensor ThermodynamicsImpl::get_mole_concentration(
    torch::Tensor yfrac) const {
  auto nvapor = options.nvapor();
  auto ncloud = options.ncloud();

  auto vec = u.sizes().vec();
  vec.erase(vec.begin());
  vec.push_back(1 + nvapor + ncloud);

  auto result = torch::empty(vec, u.options());
  auto mu = get_mu();

  result.select(3, 0) = u[index::IDN] / mu[0];
  result.narrow(3, 1, nvapor + ncloud) =
      u.narrow(0, index::ICY, nvapor + ncloud).permute({1, 2, 3, 0}) /
      mu.narrow(0, 1, nvapor + ncloud);
  return result;
}

torch::Tensor ThermodynamicsImpl::_get_internal_energy_RT(
    torch::Tensor temp) const {
  auto nvapor = options.nvapor();
  auto ncloud = options.ncloud();

  auto vec = temp.sizes().vec();
  vec.push_back(1 + nvapor + ncloud);

  auto mu = get_mu();
  auto cp = get_cp().view({1, 1, 1, -1}) * mu;
  auto result = h0 * mu + cp * (temp - options.Tref()).unsqueeze(3);
  result.narrow(3, 0, 1 + nvapor) -= constants::Rgas * temp;
  return result / (constants::Rgas * temp.unsqueeze(3));
}

}  // namespace kintera
