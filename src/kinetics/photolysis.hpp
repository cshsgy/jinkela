#pragma once

// torch
#include <torch/torch.h>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/add_arg.h>
#include <kintera/xsection/load_xsection.hpp>

// C++
#include <string>
#include <vector>
#include <map>

namespace kintera {

int locate(double const* xx, double x, int n);
void interpn(double* val, double const* coor, double const* data,
             double const* axis, size_t const* len, int ndim, int nval);

struct PhotolysisOptions {
  static PhotolysisOptions from_yaml(const YAML::Node& node);

  //! \brief wavelength grid
  //!
  //! The wavelength grid is a vector of size nwave.
  //! Default units are nanometers.
  ADD_ARG(torch::Tensor, wavelength);

  //! \brief actinic flux
  //!
  //! The actinic flux is a vector of size nwave.
  //! Default units are photons cm^-2 s^-1 nm^-1.
  ADD_ARG(torch::Tensor, actinicFlux);

  //! \brief temperature grid
};

class PhotolysisImpl : public torch::nn::Cloneable<PhotolysisImpl> {
 public:
  PhotolysisOptions options;
  PhotolysisImpl() = default;
  explicit PhotolysisImpl(PhotolysisOptions const& options_);
  void reset() override;

  //! Constructor.
  /*!
   * @param temp Temperature grid
   * @param wavelength Wavelength grid
   * @param branches Branch strings of the photolysis products
   * @param xsection Cross-section data
   */
  PhotolysisBase(vector<double> const& temp, vector<double> const& wavelength,
                 vector<std::string> const& branches,
                 vector<double> const& xsection);

  //! Constructor based on AnyValue content
  explicit PhotolysisBase(AnyMap const& node, UnitStack const& rate_units = {});

  void setParameters(YAML::Node const& node);

  void setRateParameters(YAML::Node const& rate,
                         std::map<std::string, int> const& branch_map);

  torch::Tensor getCrossSection(double temp, double wavelength) const;

 protected:
  //! composition of photolysis branch products
  std::vector<Composition> m_branch;

  //! number of temperature grid points
  size_t m_ntemp;

  //! number of wavelength grid points
  size_t m_nwave;

  //! temperature grid followed by wavelength grid
  torch::Tensor m_temp_wave_grid;

  //! \brief photolysis cross-section data
  //!
  //! The cross-section data is a three dimensional table of size (ntemp, nwave,
  //! nbranch). The first dimension is the number of temperature grid points,
  //! the second dimension is the number of wavelength grid points, and the
  //! third dimension is the number of branches of the photolysis reaction.
  //! Default units are SI units such as m, m^2, and m^2/m.
  torch::Tensor m_crossSection;
};

//! Photolysis reaction rate type depends on temperature and the actinic flux
/*!
 * A reaction rate coefficient of the following form.
 *
 * \f[
 *    k(T) = \int_{\lambda_1}^{\lambda_2} \sigma(\lambda) \phi(\lambda) d\lambda
 * \f]
 *
 * where \f$ \sigma(\lambda) \f$ is the cross-section and \f$ \phi(\lambda) \f$
 * is the actinic flux. \f$ \lambda_1 \f$ and \f$ \lambda_2 \f$ are the lower
 * and upper bounds of the wavelength grid.
 */
class PhotolysisRate : public PhotolysisBase {
 public:
  using PhotolysisBase::PhotolysisBase;  // inherit constructor

  unique_ptr<MultiRateBase> newMultiRate() const override {
    return make_unique<MultiRate<PhotolysisRate, PhotolysisData>>();
  }

  const string type() const override { return "Photolysis"; }

  Composition const& photoProducts() const override { return m_net_products; }

  double evalFromStruct(PhotolysisData const& data);

 protected:
  //! net stoichiometric coefficients of products
  Composition m_net_products;
};

}  // namespace kintera

#endif  // CT_PHOTOLYSIS_H
