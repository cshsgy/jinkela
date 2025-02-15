#pragma once

// C/C++
#include <map>
#include <string>

// kinetra
#include "add_arg.h"
#include "kinetics/ReactionRate.h"

// torch
#include <torch/torch.h>

namespace kintera {

using Composition = std::map<std::string, double>;

struct Reaction {
  Reaction() = default;
  explicit Reaction(const std::string& equation);

  Reaction(const Reaction& other)
    : rate(other.rate ? other.rate->clone() : nullptr),
      reactants_(other.reactants_),
      products_(other.products_),
      orders_(other.orders_),
      reversible_(other.reversible_) {}
  
  Reaction& operator=(const Reaction& other) {
    if (this != &other) {
      rate = other.rate ? other.rate->clone() : nullptr;
      reactants_ = other.reactants_;
      products_ = other.products_;
      orders_ = other.orders_;
      reversible_ = other.reversible_;
    }
    return *this;
  }

  std::unique_ptr<ReactionRate> rate;

  //! The chemical equation for this reaction
  std::string equation() const;

  //! Reactant species and stoichiometric coefficients
  ADD_ARG(Composition, reactants);

  //! Product species and stoichiometric coefficients
  ADD_ARG(Composition, products);

  //! Reaction orders
  ADD_ARG(Composition, orders);

  ADD_ARG(bool, reversible) = false;
};

std::string to_string(Composition const& p);

bool operator==(Reaction const& lhs, Reaction const& rhs);
bool operator<(Reaction const& lhs, Reaction const& rhs);

}  // namespace kintera
