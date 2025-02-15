#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <filesystem>

#include "kintera/utils/parse_yaml.hpp"
#include "kintera/reaction.hpp"
#include "kintera/utils/stoichiometry.hpp"
#include "kintera/kinetics/kinetics.h"

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <yaml_file>" << std::endl;
            return 1;
        }
        std::filesystem::path yaml_file = argv[1];
        
        std::vector<kintera::Reaction> reactions = kintera::parse_reactions_yaml(yaml_file.string());
        
        std::cout << "Successfully parsed " << reactions.size() << " reactions:\n\n";
        
        // Collect all unique species
        std::set<std::string> species_set;
        for (const auto& reaction : reactions) {
            for (const auto& [species, _] : reaction.reactants()) {
                species_set.insert(species);
            }
            for (const auto& [species, _] : reaction.products()) {
                species_set.insert(species);
            }
        }
        std::vector<std::string> species(species_set.begin(), species_set.end());
        
        // Generate and print stoichiometry matrix
        auto stoich_matrix = kintera::generate_stoichiometry_matrix(reactions, species);
        
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

        kintera::Kinetics kinetics(reactions, species);

        // Generate random initial conditions
        int n0 = 10;
        torch::Tensor T = torch::rand({n0});
        torch::Tensor P = torch::rand({n0});
        torch::Tensor C0 = torch::rand({n0, species.size()});

        torch::Tensor rates = torch::zeros({n0, reactions.size()});
        torch::Tensor jacobian = torch::zeros({n0, reactions.size(), species.size()});

        kinetics.eval_rates(T, P, C0, rates);
        kinetics.eval_jacobian(T, P, C0, jacobian);

        std::cout << "Rates:\n" << rates << std::endl;
        std::cout << "Jacobian:\n" << jacobian << std::endl;

        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
