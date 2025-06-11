#pragma once

// C/C++
#include <initializer_list>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/utils/func2.hpp>

#include "nucleation.hpp"

// arg
#include <kintera/add_arg.h>

namespace kintera {

//! names of all species
extern std::vector<std::string> species_names;

//! molecular weights of all species [kg/mol]
extern std::vector<double> species_weights;

template <typename T>
inline std::vector<T> insert_first(T value, std::vector<T> const& input) {
  std::vector<T> result;
  result.reserve(input.size() + 1);
  result.push_back(value);
  result.insert(result.end(), input.begin(), input.end());
  return result;
}

struct ThermoOptions {
  //! \brief Create a `ThermoOptions` object from a YAML file
  /*!
   * This function reads a YAML file and creates a `ThermoOptions`
   * object from it. The YAML file must contain the following fields:
   *  - "species", list of species names and their composition
   *  - "vapor": list of vapor species
   *  - "cloud": list of cloud species
   */
  static ThermoOptions from_yaml(std::string const& filename);

  ThermoOptions() = default;

  ADD_ARG(double, Rd) = 287.0;
  ADD_ARG(double, Tref) = 300.0;
  ADD_ARG(double, Pref) = 1.e5;

  ADD_ARG(std::vector<int>, vapor_ids);
  ADD_ARG(std::vector<int>, cloud_ids);

  ADD_ARG(std::vector<double>, mu_ratio);
  ADD_ARG(std::vector<double>, cref_R);
  ADD_ARG(std::vector<double>, uref_R);

  ADD_ARG(std::vector<user_func2>, intEng_R_extra);
  ADD_ARG(std::vector<user_func2>, cv_R_extra);
  ADD_ARG(std::vector<user_func2>, cp_R_extra);
  ADD_ARG(std::vector<user_func2>, compress_z);

  ADD_ARG(std::vector<Nucleation>, react);
  ADD_ARG(std::vector<std::string>, species);

  ADD_ARG(int, max_iter) = 5;
  ADD_ARG(double, ftol) = 1e-6;
};

//! Mass Thermodynamics
class ThermoYImpl : public torch::nn::Cloneable<ThermoYImpl> {
 public:
  //! 1. / mu. [mol/kg]
  torch::Tensor inv_mu;

  //! constant part of heat capacity at constant volume [J/(kg K)]
  torch::Tensor cv0;

  //! internal energy offset at T = 0 [J/kg]
  torch::Tensor u0;

  //! stoichiometry matrix
  torch::Tensor stoich;

  //! options with which this `ThermoY` was constructed
  ThermoOptions options;

  ThermoYImpl() = default;
  explicit ThermoYImpl(const ThermoOptions& options_);
  void reset() override;

  //! \brief perform conversions
  torch::Tensor const& compute(std::string ab,
                               std::initializer_list<torch::Tensor> args) const;

  //! \brief Perform saturation adjustment
  /*!
   * \param rho density
   * \param yfrac mass fraction
   * \param intEng zero-offset total internal energy [J/m^3]
   * \return adjusted mass fraction
   */
  torch::Tensor forward(torch::Tensor rho, torch::Tensor intEng,
                        torch::Tensor yfrac);

 private:
  //! cache
  torch::Tensor _P, _Y, _X, _V, _T, _cv, _U, _S, _F;

  //! \brief specific volume (m^3/kg) to mass fraction
  /*
   * \param[in] ivol inverse specific volume, kg/m^3, (..., 1 + ny)
   * \param[out] out mass fraction, (ny, ...)
   */
  void _ivol_to_yfrac(torch::Tensor ivol, torch::Tensor& out) const;

  //! \brief Calculate mole fraction from mass fraction
  /*!
   * Eq.77 in Li2019
   * $ x_i = \frac{y_i/\epsilon_i}{1. + \sum_{i \in V \cup C} y_i(1./\epsilon_i
   * - 1.)} $
   *
   * \param[in] yfrac mass fraction, (ny, ...)
   * \param[out] out mole fraction, (..., 1 + ny)
   */
  void _yfrac_to_xfrac(torch::Tensor yfrac, torch::Tensor& out) const;

  //! \brief mass fraction to specific volume (m^3/kg)
  /*!
   * \param[in] rho total density, kg/m^3
   * \param[in] yfrac mass fraction, (ny, ...)
   * \param[out] out inverse specific volume, kg/m^3/, (..., 1 + ny)
   */
  void _yfrac_to_ivol(torch::Tensor rho, torch::Tensor yfrac,
                      torch::Tensor& out) const;

  //! \brief Calculate temperature (K)
  /*`
   * \param[in] pres pressure, pa
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[out] temperature, K, (...)
   */
  void _pres_to_temp(torch::Tensor pres, torch::Tensor ivol,
                     torch::Tensor& out) const;

  //! \brief calculate volumetric heat capacity (J/(m^3 K))
  /*!
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[in] temp temperature, K
   * \param[out] out volumetric heat capacity, J/(m^3 K), (...)
   */
  void _cv_vol(torch::Tensor ivol, torch::Tensor temp,
               torch::Tensor& out) const;

  //! \brief calculate volumetric internal energy (J/m^3)
  /*!
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[in] temp temperature, K
   * \param[out] out volumetric internal energy, J/m^3, (...)
   */
  void _temp_to_intEng(torch::Tensor ivol, torch::Tensor temp,
                       torch::Tensor& out) const;

  //! \brief calculate temperature (K) from internal energy
  /*!
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[in] intEng volumetric internal energy, J/m^3, (...)
   * \param[out] out temperature, K, (...)
   */
  void _intEng_to_temp(torch::Tensor ivol, torch::Tensor intEng,
                       torch::Tensor& out) const;

  //! \brief calculate pressure (Pa)
  /*!
   * \param[in] ivol inverse of specific volume, kg/m^3, (..., 1 + ny)
   * \param[in] temp temperature, K
   * \param[out] out pressure, Pa, (...)
   */
  void _temp_to_pres(torch::Tensor ivol, torch::Tensor temp,
                     torch::Tensor& out) const;
};
TORCH_MODULE(ThermoY);

//! Molar thermodynamics
class ThermoXImpl : public torch::nn::Cloneable<ThermoXImpl> {
 public:
  //! mu.
  torch::Tensor mu;

  //! const part of heat capacity at constant pressure
  torch::Tensor cp0;

  //! enthalpy offset at T = 0
  torch::Tensor h0;

  //! stoichiometry matrix
  torch::Tensor stoich;

  //! options with which this `ThermoX` was constructed
  ThermoOptions options;

  ThermoXImpl() = default;
  explicit ThermoXImpl(const ThermoOptions& options_);
  void reset() override;

  //! \brief multi-component cp correction
  /*!
   * Eq.71 in Li2019
   * $ f_{\psi} = 1 + \sum_{i \in V \cup C} x_i(\sigma_{p,i} - 1) $
   *
   * \param xfrac mole fraction
   */
  torch::Tensor f_psi(torch::Tensor xfrac) const;

  //! \brief perform conversions
  torch::Tensor const& compute(std::string ab,
                               std::initializer_list<torch::Tensor> args) const;

  //! \brief Calculate the equilibrium state given temperature and pressure
  torch::Tensor forward(torch::Tensor temp, torch::Tensor pres,
                        torch::Tensor xfrac);

 private:
  //! cache
  torch::Tensor _T, _P, _X, _D, _Y, _V, _H, _S, _F, _G, _cp;

  //! \brief Calculate mass fraction from mole fraction
  /*!
   * Eq.76 in Li2019
   * $ y_i = \frac{x_i \epsilon_i}{1. + \sum_{i \in V \cup C} x_i(\epsilon_i
   * - 1.)} $
   *
   * \param xfrac mole fraction, (..., 1 + ny)
   * \return mass fraction, (ny, ...)
   */
  torch::Tensor _xfrac_to_yfrac(torch::Tensor xfrac) const;

  //! \brief Calculate concentration from mole fraction
  /*
   * $ C_i = \frac{x_i P}{R_d T_v} $
   */
  torch::Tensor _xfrac_to_conc(torch::Tensor temp, torch::Tensor pres,
                               torch::Tensor xfrac) const;

  //! \brief Calculate density from temperature, pressure and mole fraction
  /*!
   * Eq.94 in Li2019
   * $ \rho = \frac{P}{R_d T_v} $
   *
   * \param temp temperature, K
   * \param pres pressure, pa
   * \param xfrac mole fraction, (..., 1 + ny)
   */
  torch::Tensor _temp_to_dens(torch::Tensor temp, torch::Tensor pres,
                              torch::Tensor xfrac) const;

  //! \brief calculate enthalpy
  torch::Tensor _temp_to_enthalpy(
      torch::Tensor temp, torch::Tensor conc,
      torch::optional<torch::Tensor> out = torch::nullopt) const;

  //! \brief calculatec cp
  torch::Tensor _cp_mean(
      torch::Tensor temp, torch::Tensor conc,
      torch::optional<torch::Tensor> out = torch::nullopt) const;
};
TORCH_MODULE(ThermoX);

}  // namespace kintera

#undef ADD_ARG
