#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "kintera/utils/parse_yaml.hpp"
#include "kintera/reaction.hpp"

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <yaml_file>" << std::endl;
            return 1;
        }
        std::filesystem::path yaml_file = argv[1];
        
        std::vector<kintera::Reaction> reactions = kintera::parse_reactions_yaml(yaml_file.string());
        
        std::cout << "Successfully parsed " << reactions.size() << " reactions:\n\n";
        
        for (size_t i = 0; i < reactions.size(); ++i) {
            std::cout << "Reaction " << (i + 1) << ":\n";
            std::cout << "  Equation: " << reactions[i].equation() << "\n";
            
            std::cout << "  Reactants:\n";
            for (const auto& [species, coeff] : reactions[i].reactants()) {
                std::cout << "    " << species << ": " << coeff << "\n";
            }

            std::cout << "  Rate Type: " << reactions[i].rate->type() << "\n";
            std::cout << "  Rate Summary: " << reactions[i].rate->rateSummary() << "\n";
            
            std::cout << "  Products:\n";
            for (const auto& [species, coeff] : reactions[i].products()) {
                std::cout << "    " << species << ": " << coeff << "\n";
            }
            
            std::cout << "  Reversible: " << (reactions[i].reversible() ? "yes" : "no") << "\n\n";
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
