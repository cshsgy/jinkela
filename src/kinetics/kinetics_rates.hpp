#pragma once

// C++
#include <vector>
#include <string>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/reaction.hpp>

namespace kintera {

class KineticsRatesImpl : public torch::nn::Module {
 public:
  //! Constructor to initialize with reaction rates and stoichiometry matrix
  KineticsRatesImpl(const std::map<Reaction, torch::nn::AnyModule>& reaction_rates,
                    const torch::Tensor& stoich_matrix,
                    const std::vector<std::string>& species);

  //! Compute species rate of change
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [Pa], shape (...)
   * \param C concentration [kmol/m^3], shape (n_species, ...)
   * \return species rate of change [kmol/m^3/s], shape (n_species, ...)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P, torch::Tensor C) const;

 private:
  //! Map of reactions to their rate modules
  std::map<Reaction, torch::nn::AnyModule> reaction_rates_;
  
  //! Stoichiometry matrix (n_reactions x n_species)
  torch::Tensor stoich_matrix_;

  //! List of species names
  std::vector<std::string> species_;
};

TORCH_MODULE_IMPL(KineticsRates, KineticsRatesImpl);

} // namespace kintera
