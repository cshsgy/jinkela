// C/C++
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/kinetic_rate.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/kinetics/rate_constant.hpp>
#include <kintera/kinetics/species_rate.hpp>
#include <kintera/kintera_formatter.hpp>
#include <kintera/reaction.hpp>
#include <kintera/utils/parse_yaml.hpp>
#include <kintera/utils/stoichiometry.hpp>

int main(int argc, char* argv[]) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <yaml_file>" << std::endl;
      return 1;
    }
    std::filesystem::path yaml_file = argv[1];

    // rate constant
    auto rop = kintera::RateConstantOptions();
    rop.types({"Arrhenius"});
    rop.reaction_file(yaml_file.string());

    auto rate_constant = kintera::RateConstant(rop);

    // kinetic rate
    auto kop = kintera::KineticRateOptions();

    // Parse reactions from YAML
    kop.reactions() =
        kintera::parse_reactions_yaml(rop.reaction_file(), rop.types());

    std::cout << "Successfully parsed " << kop.reactions().size()
              << " reactions\n\n";

    // Collect all unique species
    std::set<std::string> species_set;
    for (const auto& reaction : kop.reactions()) {
      for (const auto& [species, _] : reaction.reactants()) {
        species_set.insert(species);
      }
      for (const auto& [species, _] : reaction.products()) {
        species_set.insert(species);
      }
    }

    kop.species() =
        std::vector<std::string>(species_set.begin(), species_set.end());

    // Create kinetics rates module
    auto kinetics = kintera::KineticRate(kop);

    int64_t ncol = 2;
    int64_t nlyr = 3;
    int64_t nspecies = static_cast<int64_t>(kop.species().size());

    // Create test conditions
    auto temp =
        300. * torch::ones({ncol, nlyr}, torch::kFloat64).requires_grad_(true);

    std::map<std::string, torch::Tensor> other;
    other["pres"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 101325.;

    auto conc = 1e-3 * torch::ones({ncol, nlyr, nspecies}, torch::kFloat64)
                           .requires_grad_(true);

    // calculate rate constant
    auto log_rc = rate_constant->forward(temp, other);
    std::cout << "log rate constant at 300 K = " << log_rc << "\n";

    // calculate kinetic rate
    kinetics->to(torch::kFloat64);
    auto reaction_rate = kinetics->forward(conc, log_rc);
    std::cout << "Reaction rates:\n" << reaction_rate << "\n\n";

    // calcualte species rates of change
    auto species_rate = kintera::species_rate(reaction_rate, kinetics->stoich);

    std::cout << "Species rates of change:\n";
    for (int64_t i = 0; i < nspecies; ++i) {
      std::cout << kop.species()[i] << ": " << species_rate.select(-1, i)
                << "\n";
    }
    std::cout << "\n";

    // Calculate Jacobian with respect to concentrations
    torch::Tensor jac =
        torch::zeros({ncol, nlyr, nspecies, nspecies}, torch::kFloat64);
    for (int64_t i = 0; i < nspecies; ++i) {
      auto grad_outputs = torch::zeros_like(species_rate);
      grad_outputs.select(-1, i) = 1.;

      auto grads = torch::autograd::grad({species_rate}, {conc}, {grad_outputs},
                                         true, true);
      // At {i, j} is the gradient of spec i with respect to spec j
      jac.select(-1, i) = grads[0];
    }

    std::cout << "Jacobian with respect to concentrations:\n";
    for (int64_t i = 0; i < nspecies; ++i) {
      for (int64_t j = 0; j < nspecies; ++j) {
        std::cout << "d(" << kop.species()[i] << ")/d(" << kop.species()[j]
                  << "): " << jac.index({0, 0, i, j}) << "\n";
      }
    }
    std::cout << "\n";

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
