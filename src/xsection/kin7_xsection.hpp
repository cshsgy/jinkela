#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
// clang-format off
#include <configure.h>
#include <add_arg.h>
// clang-format on
#include "xsection.hpp"

namespace kintera {

struct Kin7XsectionOptions {
  ADD_ARG(std::string, cross_file) = "ch4.txt";
  ADD_ARG(std::vector<std::string>, branches) = {"CH4"};
  ADD_ARG(std::vector<std::string>, species);
};

class Kin7XsectionImpl : public torch::nn::Cloneable<Kin7XsectionImpl>,
                         protected XsectionImpl {
 public:
  //! wavelength [nm]
  //! (nwave,)
  torch::Tensor kwave;

  //! photo x-section [cm^2/molecule]
  //! (nwave, nbranch)
  torch::Tensor kdata;

  //! options with which this `Kin7XsectionImpl` was constructed
  Kin7XsectionOptions options;

  //! Constructor to initialize the layer
  Kin7XsectionImpl() = default;
  explicit Kin7XsectionImpl(S8RTOptions const& options_);
  void reset() override;

  //! Get effective stoichiometric coefficients
  //! \param wave wavelength [nm], (nwave, ncol, nlyr)
  //! \param actinic flux [photons nm^-1], (nwave, ncol, nlyr)
  //! \param temp temperature [K], (ncol, nlyr)
  //! \return effective stoichiometric coeff, (ncol, nlyr, nspecies)
  torch::Tensor forward(torch::Tensor wave,
                        torch::Tensor aflux,
                        torch::optional<torch::Tensor> temp = torch::nullopt);
};
TORCH_MODULE(Kin7Xsection);

}  // namespace kintera
