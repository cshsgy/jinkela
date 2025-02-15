// C/C++
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// kintera
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

    auto reactions = kintera::parse_reactions_yaml(yaml_file.string());

    std::cout << "Successfully parsed " << reactions.size()
              << " reactions:\n\n";

    auto temp = torch::ones({2, 3}, torch::kFloat64) * 300.;
    auto pres = torch::ones({2, 3}, torch::kFloat64) * 101325.;

    for (auto& [reaction, rate] : reactions) {
      std::cout << "  Equation: " << reaction.equation() << "\n";

      std::cout << "  Reactants:\n";
      for (const auto& [species, coeff] : reaction.reactants()) {
        std::cout << "    " << species << ": " << coeff << "\n";
      }

      auto rc = rate.forward(temp, pres);
      std::cout << "rate at 300 K = " << rc << "\n";

      /*std::cout << "  Rate Type: " << rate.name() << "\n";
      std::stringstream ss;
      rate.pretty_print(ss);
      std::cout << "  Rate Summary: " << ss.str() << "\n";*/

      std::cout << "  Products:\n";
      for (const auto& [species, coeff] : reaction.products()) {
        std::cout << "    " << species << ": " << coeff << "\n";
      }

      std::cout << "  Reversible: " << (reaction.reversible() ? "yes" : "no")
                << "\n\n";
    }

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

    // Generate and print stoichiometry matrix
    auto stoich_matrix =
        kintera::generate_stoichiometry_matrix(reactions, species);

    std::cout << "\nStoichiometry Matrix:\n";
    std::cout << "Species: ";
    for (const auto& s : species) {
      std::cout << s << " ";
    }
    std::cout << "\n\n";

    for (int i = 0; i < stoich_matrix.size(0); ++i) {
      std::cout << "Reaction " << (i + 1) << ": ";
      for (int j = 0; j < stoich_matrix.size(1); ++j) {
        std::cout << stoich_matrix[i][j].item<float>() << " ";
      }
      std::cout << "\n";
    }

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
