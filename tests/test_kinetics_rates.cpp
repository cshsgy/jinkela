// C/C++
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

// kintera
#include "kintera/kinetics/kinetics_formatter.hpp"
#include "kintera/kinetics/kinetics_rates.hpp"
#include "kintera/kintera_formatter.hpp"
#include "kintera/reaction.hpp"
#include "kintera/utils/parse_yaml.hpp"
#include "kintera/utils/stoichiometry.hpp"

int main(int argc, char* argv[]) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <yaml_file>" << std::endl;
      return 1;
    }
    std::filesystem::path yaml_file = argv[1];

    // Parse reactions from YAML
    auto reactions = kintera::parse_reactions_yaml(yaml_file.string());

    std::cout << "Successfully parsed " << reactions.size()
              << " reactions:\n\n";

    // Collect all unique species
    std::set<std::string> species_set;
    for (const auto& [reaction, _] : reactions) {
      for (const auto& [species, _] : reaction.reactants()) {
        species_set.insert(species);
      }
      for (const auto& [species, _] : reaction.products()) {
        species_set.insert(species);
      }
    }
    std::vector<std::string> species(species_set.begin(), species_set.end());

    // Generate stoichiometry matrix
    auto stoich_matrix =
        kintera::generate_stoichiometry_matrix(reactions, species);

    // Create kinetics rates module
    auto kinetics = kintera::KineticsRates(reactions, stoich_matrix, species);
    
    int64_t n1 = 2;
    int64_t n2 = 3;
    int64_t n0 = static_cast<int64_t>(species.size());
    // Create test conditions
    auto temp = 300. * torch::ones({n1, n2}, torch::kFloat64).requires_grad_(true);
    auto pres = torch::ones({n1, n2}, torch::kFloat64) * 101325.;
    auto conc = 1e-3 * torch::ones({n0, n1, n2}, torch::kFloat64).requires_grad_(true);
    
    // Calculate species rates of change
    auto dcdt = kinetics->forward(temp, pres, conc);
    std::cout << "Species rates of change:\n";
    for (int64_t i = 0; i < n0; ++i) {
      std::cout << species[i] << ": " << dcdt.select(0, i) << "\n";
    }
    std::cout << "\n";

    // Calculate Jacobian with respect to concentrations
    torch::Tensor jac = torch::zeros({n0, n0, n1, n2}, torch::kFloat64);
    for (int64_t i = 0; i < n0; ++i) {
      auto grad_outputs = torch::zeros_like(dcdt);
      grad_outputs.index_put_({i}, 1.0);
      
      auto grads = torch::autograd::grad({dcdt}, {conc}, {grad_outputs}, 
                                       true, true);
      // At {i, j} is the gradient of spec i with respect to spec j
      jac.index_put_({i}, grads[0]);
    }

    std::cout << "Jacobian with respect to concentrations:\n";
    for (int64_t i = 0; i < n0; ++i) {
      for (int64_t j = 0; j < n0; ++j) {
        std::cout << "d(" << species[i] << ")/d(" << species[j] << "): "
                 << jac.index({i, j}) << "\n";
      }
    }
    std::cout << "\n";

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
} 