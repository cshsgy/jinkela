#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cctype>  
#include <cstdlib> 

#include "kintera/reaction.hpp"
#include "kintera/elements.h"

namespace kintera {

struct RateConstant {
    double A;  // pre-exponential factor
    double b;  // temperature exponent
    double Ea; // activation energy
};

// Helper function to extract a numeric value from a YAML node that may include extra text/units.
static double parse_double(const YAML::Node &node_val) {
    // Attempt to get the scalar as string regardless of whether it was originally a number.
    std::string s = node_val.as<std::string>();
    size_t start = 0;
    
    // Skip any leading whitespace.
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    
    size_t pos = start;
    // Accept valid numeric characters: digits, decimal points, exponent markers, and signs.
    while (pos < s.size() && (std::isdigit(static_cast<unsigned char>(s[pos])) ||
                              s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' ||
                              s[pos] == '+' || s[pos] == '-')) {
        ++pos;
    }
    
    std::string num_str = s.substr(start, pos - start);
    try {
        return std::stod(num_str);
    } catch (const std::exception &ex) {
        throw std::runtime_error("Failed to parse a double from string: " + s);
    }
}

RateConstant parse_rate_constant(const YAML::Node& node) {
    RateConstant rate;
    // Use the parse_double helper to allow numbers with extra unit information.
    rate.A = parse_double(node["A"]);
    rate.b = parse_double(node["b"]);
    rate.Ea = parse_double(node["Ea"]);
    return rate;
}

std::vector<Reaction> parse_reactions_yaml(const std::string& filename) {
    std::vector<Reaction> reactions;
    
    YAML::Node root = YAML::LoadFile(filename);
    printf("Loading complete\n");
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
                double pressure = parse_double(rate["P"]);  // You might want to use parse_double here too.
                RateConstant k = parse_rate_constant(rate);
            }
        }

        reactions.push_back(reaction);
    }

    return reactions;
}

} // namespace kintera
