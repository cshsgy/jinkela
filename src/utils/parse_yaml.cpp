#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cctype>  
#include <cstdlib> 

#include "../reaction.hpp"
#include "../elements.h"
#include "../kinetics/Arrhenius.h"
#include "../kinetics/ReactionRate.h"

namespace kintera {


std::vector<Reaction> parse_reactions_yaml(const std::string& filename) {
    std::vector<Reaction> reactions;
    
    YAML::Node root = YAML::LoadFile(filename);
    printf("Loading complete\n");
    for (const auto& rxn_node : root) {
        std::string equation = rxn_node["equation"].as<std::string>();
        
        Reaction reaction(equation);

        std::string type = "arrhenius";  // default type
        if (rxn_node["type"]) {
            type = rxn_node["type"].as<std::string>();
        }

        // TODO: Implement the support of other reaction types
        if (type == "arrhenius") {
            reaction.rate = std::make_unique<ArrheniusRate>(rxn_node["rate-constant"]);
        }else if (type == "three-body") {
            printf("Three-body reaction not implemented\n");
            continue;
        }else if (type == "falloff") {
            printf("Falloff reaction not implemented\n");
            continue;
        }else{
            printf("Unknown reaction type: %s\n", type.c_str());
            continue;
        }

        if (rxn_node["orders"]) {
            const auto& orders = rxn_node["orders"];
            for (const auto& order : orders) {
                std::string species = order.first.as<std::string>();
                reaction.orders()[species] = order.second.as<double>();
            }
        } else {
            for (const auto& species : reaction.reactants()) {
                reaction.orders()[species.first] = 1.0;
            }
            if (reaction.reversible()) {
                for (const auto& species : reaction.products()) {
                    reaction.orders()[species.first] = 1.0;
                }
            }
        }

        // if (rxn_node["efficiencies"]) {
        //     const auto& effs = rxn_node["efficiencies"];
        //     for (const auto& eff : effs) {
        //         std::string species = eff.first.as<std::string>();
        //         double value = eff.second.as<double>();
        //     }
        // }

        reactions.push_back(std::move(reaction));
    }

    return reactions;
}

} // namespace kintera
