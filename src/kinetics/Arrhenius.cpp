#include "Arrhenius.h"
#include "../utils/constants.h"
#include "ReactionRate.h"

#include <string>


static double parse_double(const YAML::Node &node_val) {
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
double* parse_rate_constant(const YAML::Node& node) {
    double* rate = new double[3];
    // Use the parse_double helper to allow numbers with extra unit information.
    rate[0] = parse_double(node["A"]);
    rate[1] = parse_double(node["b"]);
    rate[2] = parse_double(node["Ea"]);
    return rate;
}

namespace kintera
{

ArrheniusRate::ArrheniusRate(double A, double b, double Ea)
    : m_A(A)
    , m_b(b)
    , m_Ea_R(Ea / GasConstant)
{
    if (m_A > 0.0) {
        m_logA = std::log(m_A);
    }
}

ArrheniusRate::ArrheniusRate(const YAML::Node& node)
{
    double* rate = parse_rate_constant(node);
    m_A = rate[0];
    m_b = rate[1];
    m_Ea_R = rate[2];
    delete[] rate;
    if (m_A > 0.0) {
        m_logA = std::log(m_A);
    }
}

torch::Tensor ArrheniusRate::evalRate(torch::Tensor T, torch::Tensor P) const {
    return m_A * torch::exp(m_b * torch::log(T) - m_Ea_R * 1.0 / T);
}

torch::Tensor ArrheniusRate::ddTRate(torch::Tensor T, torch::Tensor P) const {
    return (m_Ea_R * 1.0 / T + m_b) * 1.0 / T;
}

std::string ArrheniusRate::rateSummary() const {
    return "Arrhenius Rate: A = " + std::to_string(m_A) + ", b = " + std::to_string(m_b) + ", Ea = " + std::to_string(m_Ea_R * GasConstant) + " J/mol";
}

}
