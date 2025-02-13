#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <stdexcept>

#include "kintera/reaction.hpp"
#include "kintera/elements.h"

namespace kintera {

struct RateConstant {
    double A;  // pre-exponential factor
    double b;  // temperature exponent
    double Ea; // activation energy
};

RateConstant parse_rate_constant(const YAML::Node& node) {
    RateConstant rate;
    rate.A = node["A"].as<double>();
    rate.b = node["b"].as<double>();
    rate.Ea = node["Ea"].as<double>();
    return rate;
}

std::vector<Reaction> parse_reactions_yaml(const std::string& filename) {
    std::vector<Reaction> reactions;
    
    try {
        YAML::Node root = YAML::LoadFile(filename);
        for (const auto& rxn_node : root) {
            std::string equation = rxn_node["equation"].as<std::string>();
            
            Reaction reaction(equation);

            if (rxn_node["rate-constant"]) {
                RateConstant rate = parse_rate_constant(rxn_node["rate-constant"]);
            }
            if (rxn_node["type"]) {
                std::string type = rxn_node["type"].as<std::string>();
            }

            if (rxn_node["efficiencies"]) {
                const auto& effs = rxn_node["efficiencies"];
                for (const auto& eff : effs) {
                    std::string species = eff.first.as<std::string>();
                    double value = eff.second.as<double>();
                }
            }

            if (rxn_node["orders"]) {
                const auto& orders = rxn_node["orders"];
                for (const auto& order : orders) {
                    std::string species = order.first.as<std::string>();
                    double value = order.second.as<double>();
                }
            }

            if (rxn_node["Troe"]) {
                const auto& troe = rxn_node["Troe"];
            }

            if (rxn_node["SRI"]) {
                const auto& sri = rxn_node["SRI"];
            }

            if (rxn_node["rate-constants"]) {
                const auto& plog_rates = rxn_node["rate-constants"];
                for (const auto& rate : plog_rates) {
                    double pressure = rate["P"].as<double>();
                    RateConstant k = parse_rate_constant(rate);
                }
            }

            reactions.push_back(reaction);
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing YAML file: " + std::string(e.what()));
    }

    return reactions;
}

} // namespace kintera
